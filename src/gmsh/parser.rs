use std::marker::PhantomData;

use anyhow::{Ok, Result, Error, anyhow, bail};
use winnow::{
    Parser,
    ascii::{digit1, float, line_ending, multispace0, space1},
    binary::Endianness,
    combinator::preceded,
    error::ContextError,
    token::{literal, take, take_until},
};

use crate::gmsh::mshfile::{MshDataFormat, MshFile, MshHeader, PhysicalNames};

use super::mshfile::{MshFloatType, MshIntType, MshUsizeType};

pub struct MshParser<'a, U: MshUsizeType, I: MshIntType, F: MshFloatType> {
    input: &'a [u8],
    format: Option<MshDataFormat>,
    endianness: Option<Endianness>,
    _marker: PhantomData<(U, I, F)>,
}

impl<'a, U: MshUsizeType, I: MshIntType, F: MshFloatType> MshParser<'a, U, I, F> {
    pub fn new(input: &'a [u8]) -> Self {
        Self {
            input,
            format: None,
            endianness: None,
            _marker: PhantomData,
        }
    }

    pub fn parse_bytes(&mut self) -> Result<MshFile<U, I, F>> {
        let header = self.parse_header()?;

        loop {
            self.whitespace()?;

            if self.input.is_empty() {
                break;
            }

            let (_, tag_bytes) = take_until::<_, &'a [u8], ContextError>(0.., b"\n".as_slice())
                .parse_peek(&mut self.input)
                .map_err(|e| anyhow!("failed to peek block tag: {e}"))?;

            let tag = std::str::from_utf8(tag_bytes).unwrap_or("");

            match tag {
                "$PhysicalNames" => {
                    let physical_names = self.parse_physical_names()?;
                }, 
                "$Entities" => {
                    todo!()
                }, 
                "$PartitionedEntities" => {
                    todo!()
                }, 
                "$Nodes" => {
                    todo!()
                }, 
                "$Elements" => {
                    todo!()
                }, 
                "$Periodic" => {
                    todo!()
                }, 
                "$GhostElements" => {
                    todo!()
                }, 
                "$Parametrizations" => {
                    todo!()
                }, 
                "$NodeData" => {
                    todo!()
                }, 
                "$ElementData" => {
                    todo!()
                }, 
                "$ElementNodeData" => {
                    todo!()
                }, 
                "$InterpolationScheme" => {
                    todo!()
                }
                _ => {
                    unimplemented!("unrecognised block: {}", tag)
                }
            }
        }

        Ok(MshFile { header, data: todo!() })
    }

    fn parse_header(&mut self) -> Result<MshHeader> {
        self.literal(b"$MeshFormat\n")?;

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
            preceded(space1, digit1.parse_to::<U>()),
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
        }
        
        self.format = Some(format);
        self.literal(b"$EndMeshFormat\n")?;

        Ok(MshHeader {
            version,
            format,
            data_size: data_size.to_usize().unwrap_or(8),
            endianness: self.endianness,
        })
    }

    /// Parses the $PhysicalNames block
    fn parse_physical_names(&mut self) -> Result<PhysicalNames<I>> {
        self.literal(b"$PhysicalNames")?;

        let n_physical_names = digit1::<_, ContextError>
            .parse_to::<I>()
            .parse_next(&mut self.input)
            .map_err(|e| anyhow!("failed to parse numPhysicalNames: {e}"))
            .map(|val| val.to_usize().unwrap())?;
        
        self.line_ending()?;
        
        let mut names = Vec::with_capacity(n_physical_names);
        
        for _ in 0..n_physical_names {
            let (dimension, tag, name) = (
                digit1::<_, ContextError>
                    .parse_to::<I>()
                    .map_err(|e| anyhow!("failed to parse dimension: {e}"))

            ).parse_next(&mut self.input)?;
        }

        todo!()
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
        let input = b"$MeshFormat\n4.1 0 8\n";
        let mut parser = MshParser::<usize, i32, f64>::new(input);
        let header = parser.parse_header().unwrap();

        assert_eq!(header.version, "4.1");
        assert_eq!(header.format, MshDataFormat::Ascii);
        assert_eq!(header.data_size, 8);
        assert_eq!(header.endianness, None);
    }

    #[test]
    fn test_parse_header_binary_little_endian() {
        let input = b"$MeshFormat\n4.1 1 8\n\x01\x00\x00\x00";
        let mut parser = MshParser::<usize, i32, f64>::new(input);
        let header = parser.parse_header().unwrap();

        assert_eq!(header.version, "4.1");
        assert_eq!(header.format, MshDataFormat::Binary);
        assert_eq!(header.data_size, 8);
        assert_eq!(header.endianness, Some(Endianness::Little));
    }

    #[test]
    fn test_parse_header_binary_big_endian() {
        let input = b"$MeshFormat\n4.1 1 8\n\x00\x00\x00\x01";
        let mut parser = MshParser::<usize, i32, f64>::new(input);
        let header = parser.parse_header().unwrap();

        assert_eq!(header.version, "4.1");
        assert_eq!(header.format, MshDataFormat::Binary);
        assert_eq!(header.data_size, 8);
        assert_eq!(header.endianness, Some(Endianness::Big));
    }

    #[test]
    fn test_parse_header_invalid_version() {
        let input = b"$MeshFormat\n4.2 0 8\n";
        let mut parser = MshParser::<usize, i32, f64>::new(input);
        assert!(parser.parse_header().is_err());
    }

    #[test]
    fn test_parse_header_invalid_format() {
        let input = b"$MeshFormat\n4.1 3 8\n";
        let mut parser = MshParser::<usize, i32, f64>::new(input);
        assert!(parser.parse_header().is_err());
    }
}
