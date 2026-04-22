use std::marker::PhantomData;

use anyhow::{Ok, Result, anyhow, bail};
use winnow::{
    Parser,
    ascii::{digit1, float, line_ending, space1},
    binary::Endianness,
    combinator::{opt, preceded},
    error::ContextError,
    token::{literal, take},
};

use crate::gmsh::mshfile::{MshDataFormat, MshFile, MshHeader};

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
        let header = self.parse_header().map_err(|e| anyhow!("{:?}", e))?;

        loop {
            if self.input.is_empty() {
                break;
            }
        }

        todo!()
    }

    pub fn parse_header(&mut self) -> Result<MshHeader> {
        literal::<_, _, ContextError>(b"$MeshFormat\n")
            .parse_next(&mut self.input)
            .map_err(|e| anyhow!("failed for parse mshfile header: {}", e))?;

        let (version, format, data_size) = (
            float::<_, F, ContextError>,
            preceded(
                space1,
                digit1
                    .parse_to::<i32>()
                    .verify(|&val| val == 1 || val == 2)
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

            self.endianness = match u32::from_le_bytes(
                marker
                    .try_into()
                    .map_err(|_| anyhow!("corrupt endianness marker"))?,
            ) {
                0 => Some(Endianness::Big),
                1 => Some(Endianness::Little),
                _ => bail!("corrupt endianness marker"),
            }
        }

        self.format = Some(format);

        Ok(MshHeader {
            version: version.to_string(),
            format,
            data_size: data_size.to_usize().unwrap_or(8),
            int_size: (),
            float_size: (),
            endianness: (),
        })
    }

    fn line_ending(&mut self) -> Result<&'a [u8]> {
        line_ending::<_, ContextError>
            .parse_next(&mut self.input)
            .map_err(|e| anyhow!("expected line end: {e}"))
    }
}
