use std::marker::PhantomData;

use winnow::Result;

use crate::gmsh::mshfile::MshFile;

use super::mshfile::{MshUsizeType, MshIntType, MshFloatType};

pub struct MshParser<'a, U: MshUsizeType, I: MshIntType, F: MshFloatType> {
    input: &'a [u8], 
    _marker: PhantomData<(U, I, F)>
}

impl<'a, U: MshUsizeType, I: MshIntType, F: MshFloatType> MshParser<'a, U, I, F> {
    pub fn new(input: &'a [u8]) -> Self {
        Self { input, _marker: PhantomData }
    }

    pub fn parse_bytes(&mut self) -> Result<MshFile<U, I, F>> {
        todo!()
    }
}