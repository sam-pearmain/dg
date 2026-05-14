use std::marker::PhantomData;

use anyhow::{Ok, Result, anyhow};
use winnow::{
    Parser, Stateful,
    ascii::{dec_int, multispace0, space1},
    binary::Endianness,
    combinator::{alt, delimited, preceded, repeat},
    error::ContextError,
    token::{literal, take_while},
};

use crate::gmsh::{
    helpers::{element_block, float, int, node_block, size_t, tags},
    mshfile::{
        Curve, ElementBlock, Elements, Entities, Msh, MshData, MshDataFormat, MshHeader, NodeBlock,
        Nodes, PhysicalName, PhysicalNames, Point, SizeTypeSize, Surface, Volume,
    },
};

use super::mshfile::{MshFloatType, MshIntType, MshUsizeType};

#[derive(Debug, Clone, Default)]
pub struct MshParserState {
    pub format: Option<MshDataFormat>,
    pub endianness: Option<Endianness>,
    pub size_t_size: Option<SizeTypeSize>,
}

pub type MshStream<'a> = Stateful<&'a [u8], MshParserState>;

/// The .msh file parser
pub struct MshParser<U: MshUsizeType, I: MshIntType, F: MshFloatType> {
    _marker: PhantomData<(U, I, F)>,
}

impl<'a, U: MshUsizeType, I: MshIntType, F: MshFloatType> MshParser<U, I, F> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }

    /// Top-level parse function
    pub fn parse(&self, input: &[u8]) -> Result<Msh<U, I, F>> {
        let state = MshParserState::default();
        let mut stream = Stateful { input, state };
        parse_bytes(&mut stream)
    }
}

/// Parse the .msh file bytes
fn parse_bytes<'a, U, I, F>(stream: &mut MshStream<'a>) -> Result<Msh<U, I, F>>
where
    U: MshUsizeType,
    I: MshIntType,
    F: MshFloatType,
{
    let header = parse_header(stream)?;
    let mut data = MshData::default();

    loop {
        if stream.is_empty() {
            break;
        }

        // try to get the tag for the next section
        let tag = delimited(
            multispace0,
            take_while(1.., (b'$', b'a'..=b'z', b'A'..=b'Z')),
            multispace0,
        )
        .try_map(|bytes| str::from_utf8(bytes))
        .map_err(|e: ContextError| anyhow!("failed to get section tag: {e}"))
        .parse_next(stream)?;

        match tag {
            "$PhysicalNames" => {
                let physical_names = parse_physical_names(stream)?;
                data.physical_names = Some(physical_names);
            }
            "$Entities" => {
                let entities = parse_entities::<U, I, F>(stream)?;
                data.entities = Some(entities);
            }
            "$PartitionedEntities" => {
                unimplemented!("$PartitionedEntities block is currently unsupported")
            }
            "$Nodes" => {
                let nodes = parse_nodes::<U, I, F>(stream)?;
                data.nodes = Some(nodes);
            }
            "$Elements" => {
                let elements = parse_elements::<U, I>(stream)?;
                data.elements = Some(elements);
            }
            "$Periodic" => {
                unimplemented!("$Periodic block is currently unsupported")
            }
            "$GhostElements" => {
                unimplemented!("$GhostElements block is currently unsupported")
            }
            "$Parametrizations" => {
                unimplemented!("$Parametrizations block is currently unsupported")
            }
            "$NodeData" => {
                unimplemented!("$NodeData block is currently unsupported")
            }
            "$ElementData" => {
                unimplemented!("$ElementData block is currently unsupported")
            }
            "$ElementNodeData" => {
                unimplemented!("$ElementNodeData block is currently unsupported")
            }
            "$InterpolationScheme" => {
                unimplemented!("$InterpolationScheme block is currently unsupported")
            }
            _ => {
                unimplemented!("unrecognised block: {}", tag)
            }
        }
    }

    Ok(Msh { header, data })
}

/// Parses the header block
fn parse_header<'a>(stream: &mut MshStream<'a>) -> Result<MshHeader> {
    let ((version, format, size_t_size), mut endianness) = (
        preceded(
            (literal(b"$MshFormat"), multispace0),
            (
                "4.1".value(String::from("4.1")),
                preceded(
                    space1,
                    alt((
                        "0".value(MshDataFormat::Ascii),
                        "1".value(MshDataFormat::Binary),
                    )),
                ),
                preceded(
                    space1,
                    alt(("4".value(SizeTypeSize::U32), "8".value(SizeTypeSize::U64))),
                ),
            ),
        )
        .parse_next(stream)
        .map_err(|e: ContextError| anyhow!("failed to parse mshfile header: {e}"))?,
        None,
    );

    // read the extra binary int to determine endianness
    if format == MshDataFormat::Binary {
        endianness = Some(
            delimited(
                multispace0,
                alt::<_, _, ContextError, _>((
                    [0x01, 0x00, 0x00, 0x00].value(Endianness::Little),
                    [0x00, 0x00, 0x00, 0x01].value(Endianness::Big),
                )),
                multispace0,
            )
            .parse_next(stream)
            .map_err(|e| anyhow!("corrupt or missing endianness marker: {e}"))?,
        );
    }

    // don't forget to update the state
    stream.state.format = Some(format);
    stream.state.endianness = endianness;
    stream.state.size_t_size = Some(size_t_size);

    // exit the header
    (multispace0, literal("$EndMeshFormat"), multispace0)
        .parse_next(stream)
        .map_err(|e: ContextError| anyhow!("failed to exit block: {e}"))?;

    Ok(MshHeader {
        version,
        format,
        size_t_size,
        endianness,
    })
}

/// Parses the physical names block
fn parse_physical_names<'a, I>(stream: &mut MshStream<'a>) -> Result<PhysicalNames<I>>
where
    I: MshIntType,
{
    let n_physical_names = dec_int::<_, I, _>
        .parse_next(stream)
        .map_err(|e: ContextError| anyhow!("failed to get n_physical_names: {e}"))?;

    let names: Vec<PhysicalName<I>> = repeat(
        n_physical_names.to_usize().unwrap_or_default(),
        (
            preceded(multispace0, dec_int::<_, I, _>),
            preceded(multispace0, dec_int::<_, I, _>),
            preceded(
                multispace0,
                delimited(b'"', take_while(0..=127, |c: u8| c != b'"'), b'"'),
            )
            .try_map(std::str::from_utf8)
            .map(String::from),
        )
            .map(|(dimension, tag, name)| PhysicalName {
                dimension,
                tag,
                name,
            }),
    )
    .parse_next(stream)
    .map_err(|e: ContextError| anyhow!("failed to parse physical names: {e}"))?;

    // exit the block
    (multispace0, literal("$EndPhysicalNames"), multispace0)
        .parse_next(stream)
        .map_err(|e: ContextError| anyhow!("failed to exit block: {e}"))?;

    Ok(PhysicalNames {
        n_physical_names,
        names,
    })
}

/// Parses the entities section
fn parse_entities<'a, U, I, F>(stream: &mut MshStream<'a>) -> Result<Entities<I, F>>
where
    U: MshUsizeType,
    I: MshIntType,
    F: MshFloatType,
{
    let (n_points, n_curves, n_surfaces, n_volumes) =
        (size_t::<U>, size_t::<U>, size_t::<U>, size_t::<U>)
            .parse_next(stream)
            .map_err(|e| anyhow!("failed to parse entities header: {e}"))?;

    // parse the points
    let points: Vec<Point<I, F>> = repeat(
        n_points.to_usize().unwrap_or_default(),
        (int::<I>, float::<F>, float::<F>, float::<F>, tags::<U, I>).map(
            |(tag, x, y, z, physical_tags)| Point {
                tag,
                x,
                y,
                z,
                physical_tags,
            },
        ),
    )
    .parse_next(stream)
    .map_err(|e| anyhow!("failed to parse points: {e}"))?;

    // parse the curves
    let curves: Vec<Curve<I, F>> = repeat(
        n_curves.to_usize().unwrap_or_default(),
        (
            int::<I>,
            float::<F>,
            float::<F>,
            float::<F>,
            float::<F>,
            float::<F>,
            float::<F>,
            tags::<U, I>,
            tags::<U, I>,
        )
            .map(
                |(tag, min_x, min_y, min_z, max_x, max_y, max_z, physical_tags, _)| Curve {
                    tag,
                    min_x,
                    min_y,
                    min_z,
                    max_x,
                    max_y,
                    max_z,
                    physical_tags,
                },
            ),
    )
    .parse_next(stream)
    .map_err(|e| anyhow!("failed to parse curves: {e}"))?;

    // parse the surfaces
    let surfaces: Vec<Surface<I, F>> = repeat(
        n_surfaces.to_usize().unwrap_or_default(),
        (
            int::<I>,
            float::<F>,
            float::<F>,
            float::<F>,
            float::<F>,
            float::<F>,
            float::<F>,
            tags::<U, I>,
            tags::<U, I>,
        )
            .map(
                |(tag, min_x, min_y, min_z, max_x, max_y, max_z, physical_tags, _)| Surface {
                    tag,
                    min_x,
                    min_y,
                    min_z,
                    max_x,
                    max_y,
                    max_z,
                    physical_tags,
                },
            ),
    )
    .parse_next(stream)
    .map_err(|e| anyhow!("failed to parse curves: {e}"))?;

    // parse the volumes
    let volumes: Vec<Volume<I, F>> = repeat(
        n_volumes.to_usize().unwrap_or_default(),
        (
            int::<I>,
            float::<F>,
            float::<F>,
            float::<F>,
            float::<F>,
            float::<F>,
            float::<F>,
            tags::<U, I>,
            tags::<U, I>,
        )
            .map(
                |(tag, min_x, min_y, min_z, max_x, max_y, max_z, physical_tags, _)| Volume {
                    tag,
                    min_x,
                    min_y,
                    min_z,
                    max_x,
                    max_y,
                    max_z,
                    physical_tags,
                },
            ),
    )
    .parse_next(stream)
    .map_err(|e| anyhow!("failed to parse volumes: {e}"))?;

    // exit the block
    (multispace0, literal("$EndEntities"), multispace0)
        .parse_next(stream)
        .map_err(|e: ContextError| anyhow!("failed to exit block: {e}"))?;

    Ok(Entities {
        points,
        curves,
        surfaces,
        volumes,
    })
}

/// Parses the nodes section
fn parse_nodes<'a, U, I, F>(stream: &mut MshStream<'a>) -> Result<Nodes<U, I, F>>
where
    U: MshUsizeType,
    I: MshIntType,
    F: MshFloatType,
{
    let (n_entity_blocks, n_nodes, min_node_tag, max_node_tag) =
        (size_t::<U>, size_t::<U>, size_t::<U>, size_t::<U>)
            .parse_next(stream)
            .map_err(|e| anyhow!("failed to parse node block header: {e}"))?;

    let node_blocks: Vec<NodeBlock<U, I, F>> = repeat(
        n_entity_blocks.to_usize().unwrap_or_default(),
        node_block::<U, I, F>,
    )
    .parse_next(stream)
    .map_err(|e| anyhow!("failed to parse node blocks: {e}"))?;

    // exit the block
    (multispace0, literal("$EndNodes"), multispace0)
        .parse_next(stream)
        .map_err(|e: ContextError| anyhow!("failed to exit block: {e}"))?;

    Ok(Nodes {
        n_nodes,
        min_node_tag,
        max_node_tag,
        node_blocks,
    })
}

/// Parses the elements section
fn parse_elements<'a, U, I>(stream: &mut MshStream<'a>) -> Result<Elements<U, I>>
where
    U: MshUsizeType,
    I: MshIntType,
{
    let (n_entity_blocks, n_elements, min_element_tag, max_element_tag) =
        (size_t::<U>, size_t::<U>, size_t::<U>, size_t::<U>)
            .parse_next(stream)
            .map_err(|e| anyhow!("failed to parse elements header: {e}"))?;

    let element_blocks: Vec<ElementBlock<U, I>> = repeat(
        n_entity_blocks.to_usize().unwrap_or_default(),
        element_block::<U, I>,
    )
    .parse_next(stream)
    .map_err(|e| anyhow!("failed to parse element blocks: {e}"))?;

    // exit the block
    (multispace0, literal("$EndElements"), multispace0)
        .parse_next(stream)
        .map_err(|e: ContextError| anyhow!("failed to exit block: {e}"))?;

    Ok(Elements {
        n_elements,
        min_element_tag,
        max_element_tag,
        element_blocks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use winnow::Stateful;

    // Helper to create an ASCII stream with preset state for isolated block testing
    fn create_ascii_stream(data: &[u8]) -> MshStream<'_> {
        let state = MshParserState {
            format: Some(MshDataFormat::Ascii),
            endianness: None,
            size_t_size: Some(SizeTypeSize::U64),
        };
        Stateful { input: data, state }
    }

    #[test]
    fn test_parse_header_ascii() {
        let data = b"$MshFormat\n4.1 0 8\n$EndMeshFormat\n";
        let mut stream = Stateful {
            input: data.as_ref(),
            state: MshParserState::default(),
        };

        let header = parse_header(&mut stream).unwrap();

        assert_eq!(header.version, "4.1");
        assert_eq!(header.format, MshDataFormat::Ascii);
        assert_eq!(header.size_t_size, SizeTypeSize::U64);
        assert_eq!(header.endianness, None);

        // Ensure state was correctly updated in the stream
        assert_eq!(stream.state.format, Some(MshDataFormat::Ascii));
    }

    #[test]
    fn test_parse_header_binary_le() {
        // "4.1 1 8" followed by the integer 1 in 4 bytes (Little Endian)
        let data = b"$MshFormat\n4.1 1 8\n\x01\x00\x00\x00\n$EndMeshFormat\n";
        let mut stream = Stateful {
            input: data.as_ref(),
            state: MshParserState::default(),
        };

        let header = parse_header(&mut stream).unwrap();

        assert_eq!(header.format, MshDataFormat::Binary);
        assert_eq!(header.endianness, Some(Endianness::Little));
    }

    #[test]
    fn test_parse_physical_names() {
        let data = b"2\n1 10 \"Inlet\"\n2 20 \"Wall\"\n$EndPhysicalNames\n";
        let mut stream = create_ascii_stream(data);

        let physical_names = parse_physical_names::<i32>(&mut stream).unwrap();

        assert_eq!(physical_names.n_physical_names, 2);
        assert_eq!(physical_names.names.len(), 2);

        assert_eq!(physical_names.names[0].dimension, 1);
        assert_eq!(physical_names.names[0].tag, 10);
        assert_eq!(physical_names.names[0].name, "Inlet");

        assert_eq!(physical_names.names[1].dimension, 2);
        assert_eq!(physical_names.names[1].tag, 20);
        assert_eq!(physical_names.names[1].name, "Wall");
    }

    #[test]
    fn test_parse_entities() {
        // 1 point, 1 curve, 0 surfaces, 0 volumes
        // Point: tag 1, coords (0.1, 0.2, 0.3), 1 physical tag (100)
        // Curve: tag 2, min (0,0,0), max (1,1,1), 0 physical tags, 0 bounding points
        let data = b"1 1 0 0\n1 0.1 0.2 0.3 1 100\n2 0.0 0.0 0.0 1.0 1.0 1.0 0 0\n$EndEntities\n";
        let mut stream = create_ascii_stream(data);

        let entities = parse_entities::<usize, i32, f64>(&mut stream).unwrap();

        assert_eq!(entities.points.len(), 1);
        assert_eq!(entities.points[0].tag, 1);
        assert_eq!(entities.points[0].x, 0.1);
        assert_eq!(entities.points[0].physical_tags, vec![100]);

        assert_eq!(entities.curves.len(), 1);
        assert_eq!(entities.curves[0].tag, 2);
        assert_eq!(entities.curves[0].max_x, 1.0);
        assert!(entities.curves[0].physical_tags.is_empty());
    }

    #[test]
    fn test_parse_nodes() {
        // Header: 1 entity block, 2 nodes total, min tag 1, max tag 2
        // Block 1: entity dim 0, entity tag 1, parametric 0, 2 nodes in block
        // Tags: 1, 2
        // Coords: (0,0,0), (1,0,0)
        let data = b"1 2 1 2\n0 1 0 2\n1\n2\n0.0 0.0 0.0\n1 0 0\n$EndNodes\n";
        let mut stream = create_ascii_stream(data);

        let nodes = parse_nodes::<usize, i32, f64>(&mut stream).unwrap();

        assert_eq!(nodes.n_nodes, 2);
        assert_eq!(nodes.node_blocks.len(), 1);
        assert_eq!(nodes.node_blocks[0].dim, 0);
        assert_eq!(nodes.node_blocks[0].nodes.len(), 2);
        assert_eq!(nodes.node_blocks[0].nodes[1].tag, 2);
        assert_eq!(nodes.node_blocks[0].nodes[1].x, 1.0);
    }

    #[test]
    fn test_parse_elements() {
        let data = b"1 1 10 10\n2 1 2 1\n10 1 2 3\n$EndElements\n";
        let mut stream = create_ascii_stream(data);

        let elements = parse_elements::<usize, i32>(&mut stream).unwrap();

        assert_eq!(elements.n_elements, 1);
        assert_eq!(elements.element_blocks.len(), 1);
        assert_eq!(elements.element_blocks[0].elements.len(), 1);
        assert_eq!(elements.element_blocks[0].elements[0].tag, 10);
        assert_eq!(elements.element_blocks[0].elements[0].nodes, vec![1, 2, 3]);
    }
}
