use std::marker::PhantomData;

use anyhow::{Ok, Result, anyhow};
use winnow::{
    Parser, Stateful,
    ascii::{dec_int, digit1, float, line_ending, multispace0, space1},
    binary::{Endianness, be_f64, be_i32, be_u32, be_u64, le_f64, le_i32, le_u32, le_u64},
    combinator::{alt, delimited, preceded},
    error::ContextError,
    token::{literal, take_until},
};

use crate::gmsh::mshfile::{
    DataSize, Entities, MshData, MshDataFormat, MshFile, MshHeader, PhysicalName, PhysicalNames,
};

use super::mshfile::{MshFloatType, MshIntType, MshUsizeType};

#[derive(Debug, Clone, Default)]
pub struct MshParserState {
    pub format: Option<MshDataFormat>,
    pub endianness: Option<Endianness>,
    pub usize_size: Option<DataSize>,
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

    pub fn parse(&self, input: &[u8]) -> Result<MshFile<U, I, F>> {
        let state = MshParserState::default();
        let mut stream = Stateful { input, state };
        self.parse_bytes(&mut stream)
    }

    /// Parse the .msh file bytes
    fn parse_bytes(&self, stream: &mut MshStream<'a>) -> Result<MshFile<U, I, F>> {
        let header = self.parse_header(stream)?;
        let mut data = MshData::<U, I, F> {
            physical_names: None,
            entities: None,
            nodes: None,
            elements: None,
        };

        loop {
            self.consume_whitespace(stream)?;

            if stream.is_empty() {
                break;
            }

            let (_, tag_bytes) = take_until::<_, &'a [u8], ContextError>(0.., b"\n".as_slice())
                .parse_peek(&mut self.input)
                .map_err(|e| anyhow!("failed to peek block tag: {e}"))?;

            let tag = std::str::from_utf8(tag_bytes).unwrap_or("").trim_end();

            match tag {
                "$PhysicalNames" => {
                    let physical_names = self.parse_physical_names()?;
                    data.physical_names = Some(physical_names);
                }
                "$Entities" => {
                    let entities = self.parse_entities()?;
                    data.entities = Some(entities);
                }
                "$PartitionedEntities" => {
                    todo!()
                }
                "$Nodes" => {
                    todo!()
                }
                "$Elements" => {
                    todo!()
                }
                "$Periodic" => {
                    todo!()
                }
                "$GhostElements" => {
                    todo!()
                }
                "$Parametrizations" => {
                    todo!()
                }
                "$NodeData" => {
                    todo!()
                }
                "$ElementData" => {
                    todo!()
                }
                "$ElementNodeData" => {
                    todo!()
                }
                "$InterpolationScheme" => {
                    todo!()
                }
                _ => {
                    unimplemented!("unrecognised block: {}", tag)
                }
            }
        }

        Ok(MshFile { header, data })
    }

    fn parse_header(&self, stream: &mut MshStream<'a>) -> Result<MshHeader> {
        self.consume_literal(b"$MeshFormat")?;
        self.consume_line_ending()?;

        let (version, format, data_size) = (
            float::<_, F, ContextError>
                .verify(|&version| version == F::from(4.1).unwrap())
                .map(|version| version.to_string()),
            preceded(
                space1,
                alt((
                    "0".value(MshDataFormat::Ascii),
                    "1".value(MshDataFormat::Binary),
                )),
            ),
            preceded(
                space1,
                alt(("4".value(DataSize::U32), "8".value(DataSize::U64))),
            ),
        )
            .parse_next(&mut self.input)
            .map_err(|e| anyhow!("failed to parse mshfile header: {e}"))?;

        self.consume_line_ending()?;

        if format == MshDataFormat::Binary {
            self.endianness = Some(
                alt::<_, _, ContextError, _>((
                    [0x01, 0x00, 0x00, 0x00].value(Endianness::Little),
                    [0x00, 0x00, 0x00, 0x01].value(Endianness::Big),
                ))
                .parse_next(&mut self.input)
                .map_err(|e| anyhow!("corrupt or missing endianness marker: {e}"))?,
            );

            self.consume_line_ending()?;
        }

        self.format = Some(format);
        self.consume_literal(b"$EndMeshFormat")?;
        self.consume_line_ending()?;

        Ok(MshHeader {
            version,
            format,
            data_size,
            endianness: self.endianness,
        })
    }

    /// Parses the $PhysicalNames block
    fn parse_physical_names(&mut self) -> Result<PhysicalNames<I>> {
        self.consume_literal(b"$PhysicalNames")?;
        self.consume_line_ending()?;

        let n_physical_names = digit1::<_, ContextError>
            .parse_to::<I>()
            .parse_next(&mut self.input)
            .map_err(|e| anyhow!("failed to parse numPhysicalNames: {e}"))?;

        self.consume_line_ending()?;

        let mut names = Vec::with_capacity(n_physical_names.to_usize().unwrap());

        for _ in 0..n_physical_names.to_usize().unwrap() {
            let (dimension, tag, name) = (
                digit1::<_, ContextError>.parse_to::<I>(),
                preceded(space1, digit1.parse_to::<I>()),
                preceded(
                    space1,
                    delimited(
                        b"\"",
                        take_until(0.., b"\"".as_slice())
                            .verify(|s: &[u8]| s.len() <= 127)
                            .map(|s| String::from_utf8_lossy(s).into_owned()),
                        b"\"",
                    ),
                ),
            )
                .parse_next(&mut self.input)
                .map_err(|e| anyhow!("failed to parse physical names: {e}"))?;

            names.push(PhysicalName {
                dimension,
                tag,
                name,
            });
            self.consume_line_ending()?;
        }

        self.consume_literal(b"$EndPhysicalNames")?;
        self.consume_line_ending()?;

        Ok(PhysicalNames {
            n_physical_names,
            names,
        })
    }

    /// Parses the $Entities block
    fn parse_entities(&mut self) -> Result<Entities<I, F>> {
        self.consume_literal(b"$Entities")?;
        self.consume_line_ending()?;

        let (n_points, n_curves, n_surfaces, n_volumes) = (
            self.parse_size_t(),
            self.parse_size_t(),
            self.parse_size_t(),
            self.parse_size_t(),
        )
            .parse_next(&mut self.input)
            .map_err(|e| anyhow!("erm: {e}"))?;

        todo!()
    }

    fn size_t(&self, stream: &'a mut MshStream) -> winnow::Result<U> {
        match stream.state.format.unwrap() {
            MshDataFormat::Ascii => preceded(multispace0, digit1.parse_to::<U>()),
            MshDataFormat::Binary => match (
                stream.state.endianness.unwrap(),
                stream.state.usize_size.unwrap(),
            ) {
                (Endianness::Big, DataSize::U32) => be_u32.map(|val| U::from_u32(val).unwrap()),
                (Endianness::Little, DataSize::U32) => le_u32.map(|val| U::from_u32(val).unwrap()),
                (Endianness::Big, DataSize::U64) => be_u64.map(|val| U::from_u64(val).unwrap()),
                (Endianness::Little, DataSize::U64) => le_u64.map(|val| U::from_u64(val).unwrap()),
                _ => unreachable!(),
            },
        }
        .parse_next(stream)
    }

    fn int_t(&self) -> impl Parser<&'a [u8], I, ContextError> {
        move |input: &mut &'a [u8]| match format {
            MshDataFormat::Ascii => preceded(
                multispace0,
                dec_int::<_, i32, _>.map(|val| I::from_i32(val).unwrap()),
            )
            .parse_next(input),
            MshDataFormat::Binary => match endianness {
                Endianness::Little => le_i32
                    .map(|val| I::from_i32(val).unwrap())
                    .parse_next(input),
                Endianness::Big => be_i32
                    .map(|val| I::from_i32(val).unwrap())
                    .parse_next(input),
                _ => unreachable!(),
            },
        }
    }

    fn double_t(&self, stream: &mut MshStream<'a>) -> Result<F> {
        match stream.state.format.unwrap() {
            MshDataFormat::Ascii => preceded(multispace0, float::<_, F, _>)
                .parse_next(stream)
                .map_err(|e: ContextError| anyhow!("failed to parse ascii float type: {e}")),
            MshDataFormat::Binary => match stream.state.endianness.unwrap() {
                Endianness::Big => be_f64
                    .map(|val| F::from_f64(val).unwrap())
                    .parse_next(stream)
                    .map_err(|e| anyhow!("failed to parse binary float type")),
                Endianness::Little => be_f64
                    .map(|val| F::from_f64(val).unwrap())
                    .parse_next(stream)
                    .map_err(|e| anyhow!("failed to parse binary float type")),
                _ => unreachable!(),
            },
        }
    }

    fn consume_line_ending(&self, stream: &mut MshStream<'a>) -> Result<&'a [u8]> {
        line_ending::<_, ContextError>
            .parse_next(stream)
            .map_err(|e| anyhow!("expected line end: {e}"))
    }

    fn consume_literal(&self, expected: &[u8], stream: &mut MshStream<'a>) -> Result<&'a [u8]> {
        literal::<_, _, ContextError>(expected)
            .parse_next(stream)
            .map_err(|e| anyhow!("failed to parse literal: {e}"))
    }

    fn consume_whitespace(&self, stream: &mut MshStream<'a>) -> Result<&'a [u8]> {
        multispace0::<_, ContextError>
            .parse_next(stream)
            .map_err(|e| anyhow!("failed to parse whitespace: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_header_ascii() {
        let input = b"$MeshFormat\n4.1 0 8\n$EndMeshFormat\n";
        let mut parser = MshParser::<usize, i32, f64>::new(input);
        let header = parser.parse_header().unwrap();

        assert_eq!(header.version, "4.1");
        assert_eq!(header.format, MshDataFormat::Ascii);
        assert_eq!(header.data_size, DataSize::U64);
        assert_eq!(header.endianness, None);
    }

    #[test]
    fn test_parse_header_binary_little_endian() {
        let input = b"$MeshFormat\n4.1 1 8\n\x01\x00\x00\x00\n$EndMeshFormat\n";
        let mut parser = MshParser::<usize, i32, f64>::new(input);
        let header = parser.parse_header().unwrap();

        assert_eq!(header.version, "4.1");
        assert_eq!(header.format, MshDataFormat::Binary);
        assert_eq!(header.data_size, DataSize::U64);
        assert_eq!(header.endianness, Some(Endianness::Little));
    }

    #[test]
    fn test_parse_header_binary_big_endian() {
        let input = b"$MeshFormat\n4.1 1 8\n\x00\x00\x00\x01\n$EndMeshFormat\n";
        let mut parser = MshParser::<usize, i32, f64>::new(input);
        let header = parser.parse_header().unwrap();

        assert_eq!(header.version, "4.1");
        assert_eq!(header.format, MshDataFormat::Binary);
        assert_eq!(header.data_size, DataSize::U64);
        assert_eq!(header.endianness, Some(Endianness::Big));
    }

    #[test]
    fn test_parse_header_invalid_version() {
        let input = b"$MeshFormat\n4.2 0 8\n$EndMeshFormat\n";
        let mut parser = MshParser::<usize, i32, f64>::new(input);
        assert!(parser.parse_header().is_err());
    }

    #[test]
    fn test_parse_header_invalid_format() {
        let input = b"$MeshFormat\n4.1 3 8\n$EndMeshFormat\n";
        let mut parser = MshParser::<usize, i32, f64>::new(input);
        assert!(parser.parse_header().is_err());
    }
}
