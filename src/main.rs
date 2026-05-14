use winnow::Parser;
use winnow::Result;
use winnow::error::ParserError;
use winnow::stream::Stream;

pub fn parse_prefix(input: &mut &str) -> Result<char> {
    let c = input
        .next_token()
        .ok_or_else(|| ParserError::from_input(input))?;

    if c != '0' {
        return Err(ParserError::from_input(input));
    }

    Ok(c)
}

fn main() {
    let mut input = "0x1a2b hi";

    // let output = parse_prefix(&mut input).unwrap();
    let same_thing = parse_prefix
        .verify(|c| c.is_ascii())
        .parse_next(&mut input)
        .unwrap();

    assert_eq!(input, "x1a2b hi");
    // assert_eq!(output, '0');
}
