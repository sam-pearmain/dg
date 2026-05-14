use std::marker::PhantomData;

use anyhow::{Ok, Result, anyhow};
use winnow::{
    Parser, Stateful,
    ascii::{dec_int, multispace0, space1},
    binary::Endianness,
    combinator::{alt, delimited, preceded, repeat, terminated},
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

impl MshParserState {
    fn is_ascii(&self) -> Option<bool> {
        if let Some(format) = self.format {
            Some(format == MshDataFormat::Ascii)
        } else {
            None
        }
    }

    fn is_binary(&self) -> Option<bool> {
        if let Some(format) = self.format {
            Some(format == MshDataFormat::Binary)
        } else {
            None
        }
    }

    fn is_binary_le(&self) -> Option<bool> {
        if let Some(is_binary) = self.is_binary() {
            if let Some(endianness) = self.endianness {
                Some(is_binary && endianness == Endianness::Little)
            } else {
                None
            }
        } else {
            None
        }
    }

    fn is_binary_be(&self) -> Option<bool> {
        if let Some(is_binary) = self.is_binary() {
            if let Some(endianness) = self.endianness {
                Some(is_binary && endianness == Endianness::Big)
            } else {
                None
            }
        } else {
            None
        }
    }
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
            terminated(
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

    Ok(Elements {
        n_elements,
        min_element_tag,
        max_element_tag,
        element_blocks,
    })
}
