use std::marker::PhantomData;

use anyhow::{Ok, Result, anyhow};
use winnow::{
    Parser, Stateful,
    ascii::{dec_int, digit1, float, line_ending, multispace0, newline, space1},
    binary::{Endianness, be_f64, be_i32, be_u32, be_u64, le_f64, le_i32, le_u32, le_u64},
    combinator::{alt, delimited, dispatch, fail, preceded, repeat, terminated},
    error::ContextError,
    prelude::*,
    token::{literal, take_until, take_while},
};

use crate::gmsh::mshfile::{
    Entities, MshData, MshDataFormat, MshFile, MshHeader, PhysicalName, PhysicalNames, SizeTypeSize,
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
    pub fn parse(&self, input: &[u8]) -> Result<MshFile<U, I, F>> {
        let state = MshParserState::default();
        let mut stream = Stateful { input, state };
        self.parse_bytes(&mut stream)
    }

    /// Parse the .msh file bytes
    fn parse_bytes(&self, stream: &mut MshStream<'a>) -> Result<MshFile<U, I, F>> {
        let header = self.parse_header(stream)?;
        let mut data = MshData::default();

        loop {
            if stream.is_empty() {
                break;
            }

            // try to get the tag for the next section
            let tag = terminated(
                take_while(0.., (b'$', b'a'..=b'z', b'A'..=b'Z')),
                multispace0,
            )
            .try_map(|bytes| str::from_utf8(bytes))
            .map_err(|e: ContextError| anyhow!("failed to get section tag: {e}"))
            .parse_next(stream)?;

            match tag {
                "$PhysicalNames" => {
                    todo!()
                }
                "$Entities" => {
                    todo!()
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

    /// Parses the header block
    fn parse_header(&self, stream: &mut MshStream<'a>) -> Result<MshHeader> {
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
}
