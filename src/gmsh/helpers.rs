use num::FromPrimitive;
use winnow::{
    Parser, Result,
    ascii::{dec_int, dec_uint, multispace0},
    binary::{Endianness, be_f64, be_i32, be_u32, be_u64, le_f64, le_i32, le_u32, le_u64},
    combinator::{fail, preceded, repeat},
};

use crate::gmsh::{
    mshfile::{
        Element, ElementBlock, GmshElementType, MshDataFormat, MshFloatType, MshIntType,
        MshUsizeType, Node, NodeBlock, SizeTypeSize,
    },
    parsers::MshStream,
};

/// Parses a msh size_t type
pub fn size_t<'a, U: MshUsizeType>(stream: &mut MshStream<'a>) -> Result<U> {
    match stream.state.format {
        Some(MshDataFormat::Ascii) => preceded(multispace0, dec_uint::<_, U, _>).parse_next(stream),
        Some(MshDataFormat::Binary) => match (stream.state.size_t_size, stream.state.endianness) {
            (Some(SizeTypeSize::U32), Some(Endianness::Little)) => {
                le_u32.verify_map(U::from_u32).parse_next(stream)
            }
            (Some(SizeTypeSize::U32), Some(Endianness::Big)) => {
                be_u32.verify_map(U::from_u32).parse_next(stream)
            }
            (Some(SizeTypeSize::U64), Some(Endianness::Little)) => {
                le_u64.verify_map(U::from_u64).parse_next(stream)
            }
            (Some(SizeTypeSize::U64), Some(Endianness::Big)) => {
                be_u64.verify_map(U::from_u64).parse_next(stream)
            }
            _ => fail.parse_next(stream),
        },
        None => fail.parse_next(stream),
    }
}

/// Parses a msh int type
pub fn int<'a, I: MshIntType>(stream: &mut MshStream<'a>) -> Result<I> {
    match stream.state.format {
        Some(MshDataFormat::Ascii) => preceded(multispace0, dec_int::<_, I, _>).parse_next(stream),
        Some(MshDataFormat::Binary) => match stream.state.endianness {
            Some(Endianness::Little) => le_i32.verify_map(I::from_i32).parse_next(stream),
            Some(Endianness::Big) => be_i32.verify_map(I::from_i32).parse_next(stream),
            _ => fail.parse_next(stream),
        },
        None => fail.parse_next(stream),
    }
}

/// Parses a msh float type
pub fn float<'a, F: MshFloatType>(stream: &mut MshStream<'a>) -> Result<F> {
    match stream.state.format {
        Some(MshDataFormat::Ascii) => {
            preceded(multispace0, winnow::ascii::float).parse_next(stream)
        }
        Some(MshDataFormat::Binary) => match stream.state.endianness {
            Some(Endianness::Little) => le_f64.verify_map(F::from_f64).parse_next(stream),
            Some(Endianness::Big) => be_f64.verify_map(F::from_f64).parse_next(stream),
            _ => fail.parse_next(stream),
        },
        None => fail.parse_next(stream),
    }
}

/// Parses a list of tags preceded by a size_t
pub fn tags<'a, U: MshUsizeType, I: MshIntType>(stream: &mut MshStream<'a>) -> Result<Vec<I>> {
    let count = size_t::<U>(stream)?;
    repeat(count.to_usize().unwrap_or_default(), int::<I>).parse_next(stream)
}

/// Parses a node block
pub fn node_block<'a, U, I, F>(stream: &mut MshStream<'a>) -> Result<NodeBlock<U, I, F>>
where
    U: MshUsizeType,
    I: MshIntType,
    F: MshFloatType,
{
    // fail if we detect parametric
    let (dim, tag, _parametric, n_nodes) = (
        int::<I>,
        int::<I>,
        int::<I>.verify(|v| *v == I::zero()),
        size_t::<U>,
    )
        .parse_next(stream)?;

    let tags: Vec<U> =
        repeat(n_nodes.to_usize().unwrap_or_default(), size_t::<U>).parse_next(stream)?;

    let xyz: Vec<(F, F, F)> = repeat(
        n_nodes.to_usize().unwrap_or_default(),
        (float::<F>, float::<F>, float::<F>),
    )
    .parse_next(stream)?;

    let nodes: Vec<Node<U, F>> = tags
        .into_iter()
        .zip(xyz)
        .map(|(tag, (x, y, z))| Node { tag, x, y, z })
        .collect();

    Ok(NodeBlock { dim, tag, nodes })
}

/// Parses an element block
pub fn element_block<'a, U, I>(stream: &mut MshStream<'a>) -> Result<ElementBlock<U, I>>
where
    U: MshUsizeType,
    I: MshIntType,
{
    let (dim, tag, element_type, n_elements_in_block) =
        (int::<I>, int::<I>, element_type::<I>, size_t::<U>).parse_next(stream)?;

    let elements: Vec<Element<U>> = repeat(
        n_elements_in_block.to_usize().unwrap_or_default(),
        element::<U>(element_type.n_nodes().unwrap()),
    )
    .parse_next(stream)?;

    Ok(ElementBlock {
        dim,
        tag,
        element_type,
        elements,
    })
}

/// Parses the element type
pub fn element_type<'a, I>(stream: &mut MshStream<'a>) -> Result<GmshElementType>
where
    I: MshIntType,
{
    int::<I>
        .verify_map(|v| GmshElementType::from_i32(v.to_i32()?))
        .parse_next(stream)
}

/// Parses an element
pub fn element<'a, U>(n_nodes: usize) -> impl FnMut(&mut MshStream<'a>) -> Result<Element<U>>
where
    U: MshUsizeType,
{
    move |stream: &mut MshStream<'a>| {
        let tag = size_t::<U>.parse_next(stream)?;
        let nodes: Vec<U> = repeat(n_nodes, size_t::<U>).parse_next(stream)?;

        Ok(Element { tag, nodes })
    }
}
