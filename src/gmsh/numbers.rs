use winnow::{
    Parser, Result,
    ascii::{dec_int, dec_uint},
    binary::{Endianness, be_f64, be_i32, be_u32, be_u64, le_f64, le_i32, le_u32, le_u64},
    combinator::fail,
    error::ContextError,
};

use crate::gmsh::{
    mshfile::{MshDataFormat, MshFloatType, MshIntType, MshUsizeType, SizeTypeSize},
    parser::MshStream,
};

/// Parses a msh size_t type
pub fn size_t<'a, U: MshUsizeType>(input: &mut MshStream<'a>) -> Result<U> {
    match input.state.format {
        Some(MshDataFormat::Ascii) => dec_uint::<_, U, ContextError>.parse_next(input),
        Some(MshDataFormat::Binary) => match (input.state.size_t_size, input.state.endianness) {
            (Some(SizeTypeSize::U32), Some(Endianness::Little)) => {
                le_u32.verify_map(U::from_u32).parse_next(input)
            }
            (Some(SizeTypeSize::U32), Some(Endianness::Big)) => {
                be_u32.verify_map(U::from_u32).parse_next(input)
            }
            (Some(SizeTypeSize::U64), Some(Endianness::Little)) => {
                le_u64.verify_map(U::from_u64).parse_next(input)
            }
            (Some(SizeTypeSize::U64), Some(Endianness::Big)) => {
                be_u64.verify_map(U::from_u64).parse_next(input)
            }
            _ => fail.parse_next(input),
        },
        None => fail.parse_next(input),
    }
}

/// Parses a msh int type
pub fn int<'a, I: MshIntType>(input: &mut MshStream<'a>) -> Result<I> {
    match input.state.format {
        Some(MshDataFormat::Ascii) => dec_int::<_, I, ContextError>.parse_next(input),
        Some(MshDataFormat::Binary) => match input.state.endianness {
            Some(Endianness::Little) => le_i32.verify_map(I::from_i32).parse_next(input),
            Some(Endianness::Big) => be_i32.verify_map(I::from_i32).parse_next(input),
            _ => fail.parse_next(input),
        },
        None => fail.parse_next(input),
    }
}

/// Parses a msh float type
pub fn float<'a, F: MshFloatType>(input: &mut MshStream<'a>) -> Result<F> {
    match input.state.format {
        Some(MshDataFormat::Ascii) => winnow::ascii::float.parse_next(input),
        Some(MshDataFormat::Binary) => match input.state.endianness {
            Some(Endianness::Little) => le_f64.verify_map(F::from_f64).parse_next(input),
            Some(Endianness::Big) => be_f64.verify_map(F::from_f64).parse_next(input),
            _ => fail.parse_next(input),
        },
        None => fail.parse_next(input),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::anyhow;
    use winnow::{Stateful, ascii::multispace0, combinator::preceded};

    use crate::gmsh::parser::MshParserState;

    #[test]
    fn test_ascii_primitives() {
        let input_data = b"42 -123 3.14159";
        let mut stream = Stateful {
            input: &input_data[..],
            state: MshParserState {
                format: Some(MshDataFormat::Ascii),
                ..Default::default()
            },
        };

        let (s, i, f) = (
            size_t::<usize>,
            preceded(multispace0, int::<i32>),
            preceded(multispace0, float::<f64>),
        )
            .parse_next(&mut stream)
            .map_err(|_| anyhow!("erm"))
            .expect("wtf");

        assert_eq!(s, 42);
        assert_eq!(i, -123);
        assert!((f - 3.14159).abs() < f64::EPSILON);
    }

    #[test]
    fn test_binary_le_primitives() {
        let mut input_data = Vec::new();
        input_data.extend_from_slice(&1000u64.to_le_bytes());
        input_data.extend_from_slice(&(-50i32).to_le_bytes());
        input_data.extend_from_slice(&2.718f64.to_le_bytes());

        let mut stream = Stateful {
            input: &input_data[..],
            state: MshParserState {
                format: Some(MshDataFormat::Binary),
                endianness: Some(Endianness::Little),
                size_t_size: Some(SizeTypeSize::U64),
            },
        };

        let s: usize = size_t(&mut stream).unwrap();
        assert_eq!(s, 1000);

        let i: i32 = int(&mut stream).unwrap();
        assert_eq!(i, -50);

        let f: f64 = float(&mut stream).unwrap();
        assert!((f - 2.718).abs() < f64::EPSILON);
    }

    #[test]
    fn test_binary_le_u32_size_t() {
        let input_data = 500u32.to_le_bytes();
        let mut stream = Stateful {
            input: &input_data[..],
            state: MshParserState {
                format: Some(MshDataFormat::Binary),
                endianness: Some(Endianness::Little),
                size_t_size: Some(SizeTypeSize::U32),
            },
        };

        let s: usize = size_t(&mut stream).unwrap();
        assert_eq!(s, 500);
    }

    #[test]
    fn test_binary_be_primitives() {
        let mut input_data = Vec::new();
        input_data.extend_from_slice(&100u32.to_be_bytes());
        input_data.extend_from_slice(&12345i32.to_be_bytes());
        input_data.extend_from_slice(&1.414f64.to_be_bytes());

        let mut stream = Stateful {
            input: &input_data[..],
            state: MshParserState {
                format: Some(MshDataFormat::Binary),
                endianness: Some(Endianness::Big),
                size_t_size: Some(SizeTypeSize::U32),
            },
        };

        let s: usize = size_t(&mut stream).unwrap();
        assert_eq!(s, 100);

        let i: i32 = int(&mut stream).unwrap();
        assert_eq!(i, 12345);

        let f: f64 = float(&mut stream).unwrap();
        assert!((f - 1.414).abs() < f64::EPSILON);
    }

    #[test]
    fn test_missing_state_errors() {
        let input_data = b"any";
        let mut stream = Stateful {
            input: &input_data[..],
            state: MshParserState {
                format: None, // Missing format should trigger 'fail'
                ..Default::default()
            },
        };

        assert!(size_t::<usize>(&mut stream).is_err());
        assert!(int::<i32>(&mut stream).is_err());
        assert!(float::<f64>(&mut stream).is_err());
    }

    #[test]
    fn test_combinator_integration() {
        use winnow::ascii::multispace1;
        use winnow::combinator::preceded;

        let input_data = b"  -999";
        let mut stream = Stateful {
            input: &input_data[..],
            state: MshParserState {
                format: Some(MshDataFormat::Ascii),
                ..Default::default()
            },
        };

        let result: i32 = preceded(multispace1, int).parse_next(&mut stream).unwrap();
        assert_eq!(result, -999);
    }
}
