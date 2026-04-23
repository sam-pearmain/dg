use std::marker::PhantomData;

use anyhow::{Error, Ok, Result, anyhow, bail};
use winnow::{
    Parser,
    ascii::{digit1, float, line_ending, multispace0, space1},
    binary::{Endianness, be_u32, be_u64, le_u32, le_u64},
    combinator::{delimited, preceded},
    error::ContextError,
    token::{literal, take, take_until},
};

use crate::gmsh::mshfile::{
    DataSize, Entities, MshData, MshDataFormat, MshFile, MshHeader, PhysicalName, PhysicalNames,
};

use super::mshfile::{MshFloatType, MshIntType, MshUsizeType};

/// The .msh file parser
pub struct MshParser<'a, U: MshUsizeType, I: MshIntType, F: MshFloatType> {
    input: &'a [u8],
    format: Option<MshDataFormat>,
    data_size: Option<DataSize>,
    endianness: Option<Endianness>,
    _marker: PhantomData<(U, I, F)>,
}

impl<'a, U: MshUsizeType, I: MshIntType, F: MshFloatType> MshParser<'a, U, I, F> {
    pub fn new(input: &'a [u8]) -> Self {
        Self {
            input,
            format: None,
            data_size: None,
            endianness: None,
            _marker: PhantomData,
        }
    }

    /// The top level parser which parses every byte as read from the .msh file
    pub fn parse_bytes(&mut self) -> Result<MshFile<U, I, F>> {
        let header = self.parse_header()?;
        let mut data = MshData::<U, I, F> {
            physical_names: None,
            entities: None,
            nodes: None,
            elements: None,
        };

        loop {
            self.whitespace()?;

            if self.input.is_empty() {
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

    fn parse_header(&mut self) -> Result<MshHeader> {
        self.literal(b"$MeshFormat")?;
        self.line_ending()?;

        let (version, format, data_size) = (
            float::<_, F, ContextError>
                .verify(|&version| version == F::from(4.1).unwrap())
                .map(|version| version.to_string()),
            preceded(
                space1,
                digit1
                    .parse_to::<i32>()
                    .verify(|&val| val == 0 || val == 1)
                    .map(|val| {
                        if val == 0 {
                            MshDataFormat::Ascii
                        } else {
                            MshDataFormat::Binary
                        }
                    }),
            ),
            preceded(
                space1,
                digit1
                    .parse_to::<usize>()
                    .verify(|&val| val == 4usize || val == 8usize)
                    .map(|val| {
                        if val == 4 {
                            DataSize::U32
                        } else {
                            // we can be confident here since we have already checked if val == 4 or val == 8
                            DataSize::U64
                        }
                    }),
            ),
        )
            .parse_next(&mut self.input)
            .map_err(|e| anyhow!("failed to parse mshfile header: {e}"))?;

        self.line_ending()?;

        if format == MshDataFormat::Binary {
            let marker: &'a [u8] = take::<_, _, ContextError>(4usize)
                .parse_next(&mut self.input)
                .map_err(|e| anyhow!("failed to parse endianness marker: {e}"))?;

            self.endianness = match marker {
                [0x01, 0x00, 0x00, 0x00] => Some(Endianness::Little),
                [0x00, 0x00, 0x00, 0x01] => Some(Endianness::Big),
                _ => bail!("corrupt endianness marker"),
            };

            self.line_ending()?;
        }

        self.format = Some(format);
        self.literal(b"$EndMeshFormat")?;
        self.line_ending()?;

        Ok(MshHeader {
            version,
            format,
            data_size,
            endianness: self.endianness,
        })
    }

    /// Parses the $PhysicalNames block
    fn parse_physical_names(&mut self) -> Result<PhysicalNames<I>> {
        self.literal(b"$PhysicalNames")?;
        self.line_ending()?;

        let n_physical_names = digit1::<_, ContextError>
            .parse_to::<I>()
            .parse_next(&mut self.input)
            .map_err(|e| anyhow!("failed to parse numPhysicalNames: {e}"))?;

        self.line_ending()?;

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
            self.line_ending()?;
        }

        self.literal(b"$EndPhysicalNames")?;
        self.line_ending()?;

        Ok(PhysicalNames {
            n_physical_names,
            names,
        })
    }

    /// Parses the $Entities block
    fn parse_entities(&mut self) -> Result<Entities<I, F>> {
        self.literal(b"$Entities")?;
        self.line_ending()?;

        todo!()
    }

    /// Parses a size_t type
    fn size_t(&mut self) -> Result<U> {
        match self.format.unwrap() {
            MshDataFormat::Ascii => digit1::<_, ContextError>
                .parse_to::<U>()
                .parse_next(&mut self.input)
                .map_err(|e| anyhow!("failed to parse ascii size_t: {e}")),
            MshDataFormat::Binary => match self.data_size.unwrap() {
                DataSize::U32 => match self.endianness.unwrap() {
                    Endianness::Big => be_u32::<_, ContextError>
                        .parse_next(&mut self.input)
                        .map_err(|e| anyhow!("failed to parse be_u32: {e}"))
                        .and_then(|val| {
                            U::from_u32(val)
                                .ok_or_else(|| anyhow!("failed to convert u32 to generic type"))
                        }),
                    Endianness::Little => le_u32::<_, ContextError>
                        .parse_next(&mut self.input)
                        .map_err(|e| anyhow!("failed to parse le_u32: {e}"))
                        .and_then(|val| {
                            U::from_u32(val)
                                .ok_or_else(|| anyhow!("failed to convert u32 to generic type"))
                        }),
                    _ => bail!("this should not happen"),
                },
                DataSize::U64 => match self.endianness.unwrap() {
                    Endianness::Big => be_u64::<_, ContextError>
                        .parse_next(&mut self.input)
                        .map_err(|e| anyhow!("failed to parse be_u64: {e}"))
                        .and_then(|val| {
                            U::from_u64(val)
                                .ok_or_else(|| anyhow!("failed to convert u64 to generic type"))
                        }),
                    Endianness::Little => le_u64::<_, ContextError>
                        .parse_next(&mut self.input)
                        .map_err(|e| anyhow!("failed to parse le_u64: {e}"))
                        .and_then(|val| {
                            U::from_u64(val)
                                .ok_or_else(|| anyhow!("failed to convert u64 to generic type"))
                        }),
                    _ => bail!("this should not happen"),
                },
            },
        }
    }

    /// Consumes newlines
    fn line_ending(&mut self) -> Result<&'a [u8]> {
        line_ending::<_, ContextError>
            .parse_next(&mut self.input)
            .map_err(|e| anyhow!("expected line end: {e}"))
    }

    /// Consume a literal
    fn literal(&mut self, expected: &[u8]) -> Result<&'a [u8]> {
        literal::<_, _, ContextError>(expected)
            .parse_next(&mut self.input)
            .map_err(|e| anyhow!("failed to parse literal: {e}"))
    }

    /// Consumes whitespace
    fn whitespace(&mut self) -> Result<&'a [u8]> {
        multispace0::<_, ContextError>
            .parse_next(&mut self.input)
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
