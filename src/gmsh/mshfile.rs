use std::{collections::HashMap, hash::Hash};

use anyhow::{Ok, Result, bail};
use ndarray::{Array1, Array2};
use num::{Float, FromPrimitive, Integer, Signed};
use num_traits::Unsigned;

/// Trait for msh file usize_t types
pub trait MshUsizeType: Unsigned + Integer + Clone + Copy + Hash + FromPrimitive {}

/// Trait for msh file int types
pub trait MshIntType: Signed + Integer + Clone + Copy + Hash + FromPrimitive {}

/// Trait for msh file float types
pub trait MshFloatType: Float + Clone + Copy + Hash + FromPrimitive {}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Endianness {
    Big,
    Little,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MshFileType {
    Ascii,
    Binary,
}

pub struct MshFile<U: MshUsizeType, I: MshIntType, F: MshFloatType> {
    pub header: MshHeader,
    pub data: MshData<U, I, F>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MshHeader {
    pub version: String,
    pub file_type: i32,
    pub size_t_size: usize,
    pub int_size: usize,
    pub float_size: usize,
    pub endianness: Option<Endianness>,
}

pub struct MshData<U: MshUsizeType, I: MshIntType, F: MshFloatType> {
    pub entities: Option<Entities<I, F>>,
    pub nodes: Option<Nodes<U, I, F>>,
    pub elements: Option<Elements<U, I>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Entities<I: MshIntType, F: MshFloatType> {
    pub points: Vec<Point<I, F>>,
    pub curves: Vec<Curve<I, F>>,
    pub surfaces: Vec<Surface<I, F>>,
    pub volumes: Vec<Volume<I, F>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Point<I: MshIntType, F: MshFloatType> {
    pub tag: I,
    pub x: F,
    pub y: F,
    pub z: F,
    pub physical_tags: Vec<I>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Curve<I: MshIntType, F: MshFloatType> {
    pub tag: I,
    pub min_x: F,
    pub min_y: F,
    pub min_z: F,
    pub max_x: F,
    pub max_y: F,
    pub max_z: F,
    pub point_tags: Vec<I>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Surface<I: MshIntType, F: MshFloatType> {
    pub tag: I,
    pub min_x: F,
    pub min_y: F,
    pub min_z: F,
    pub max_x: F,
    pub max_y: F,
    pub max_z: F,
    pub curve_tags: Vec<I>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Volume<I: MshIntType, F: MshFloatType> {
    pub tag: I,
    pub min_x: F,
    pub min_y: F,
    pub min_z: F,
    pub max_x: F,
    pub max_y: F,
    pub max_z: F,
    pub curve_tags: Vec<I>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Nodes<U: MshUsizeType, I: MshIntType, F: MshFloatType> {
    pub n_nodes: U,
    pub min_node_tag: U,
    pub max_node_tag: U,
    pub blocks: Vec<NodeBlock<U, I, F>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NodeBlock<U: MshUsizeType, I: MshIntType, F: MshFloatType> {
    pub entity_dim: I,
    pub entity_tag: I,
    pub node_tags: Option<HashMap<U, usize>>,
    pub nodes: Array2<F>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Elements<U: MshUsizeType, I: MshIntType> {
    pub num_elements: U,
    pub min_element_tag: U,
    pub max_element_tag: U,
    pub element_blocks: Vec<ElementBlock<U, I>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ElementBlock<U: MshUsizeType, I: MshIntType> {
    pub entity_dim: I,
    pub entity_tag: I,
    pub element_type: GmshElementType,
    pub element_tags: Option<HashMap<U, usize>>,
    pub elements: Vec<Element<U>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Element<U: MshUsizeType> {
    pub tag: U,
    pub nodes: Array1<U>,
}

#[rustfmt::skip]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum GmshElementType {
    Pnt = 15, PntSub = 133,
    Lin1 = 84, Lin2 = 1, Lin3 = 8, Lin4 = 26, Lin5 = 27, Lin6 = 28, Lin7 = 62, Lin8 = 63, Lin9 = 64, Lin10 = 65, Lin11 = 66, LinB = 67, LinC = 70, LinSub = 134,
    Tri1 = 85, Tri3 = 2, Tri6 = 9, Tri9 = 20, Tri10 = 21, Tri12 = 22, Tri15 = 23, Tri15i = 24, Tri18 = 52, Tri21 = 25, Tri21i = 53, Tri24 = 54, Tri27 = 55, Tri28 = 42, Tri30 = 56, Tri36 = 43, Tri45 = 44, Tri55 = 45, Tri66 = 46, TriB = 68, TriMini = 138, TriSub = 135, Trih4 = 140,
    Qua1 = 86, Qua4 = 3, Qua8 = 16, Qua9 = 10, Qua12 = 39, Qua16 = 36, Qua16i = 40, Qua20 = 41, Qua24 = 57, Qua25 = 37, Qua28 = 58, Qua32 = 59, Qua36 = 38, Qua36i = 60, Qua40 = 61, Qua49 = 47, Qua64 = 48, Qua81 = 49, Qua100 = 50, Qua121 = 51,
    Polyg = 34, PolygB = 69, Polyh = 35,
    Tet1 = 87, Tet4 = 4, Tet10 = 11, Tet16 = 137, Tet20 = 29, Tet22 = 32, Tet28 = 33, Tet34 = 79, Tet35 = 30, Tet40 = 80, Tet46 = 81, Tet52 = 82, Tet56 = 31, Tet58 = 83, Tet84 = 71, Tet120 = 72, Tet165 = 73, Tet220 = 74, Tet286 = 75, TetMini = 139, TetSub = 136,
    Hex1 = 88, Hex8 = 5, Hex20 = 17, Hex27 = 12, Hex32 = 99, Hex44 = 100, Hex56 = 101, Hex64 = 92, Hex68 = 102, Hex80 = 103, Hex92 = 104, Hex104 = 105, Hex125 = 93, Hex216 = 94, Hex343 = 95, Hex512 = 96, Hex729 = 97, Hex1000 = 98,
    Pri1 = 89, Pri6 = 6, Pri15 = 18, Pri18 = 13, Pri24 = 111, Pri33 = 112, Pri40 = 90, Pri42 = 113, Pri51 = 114, Pri60 = 115, Pri69 = 116, Pri75 = 91, Pri78 = 117, Pri126 = 106, Pri196 = 107, Pri288 = 108, Pri405 = 109, Pri550 = 110,
    Pyr1 = 132, Pyr5 = 7, Pyr13 = 19, Pyr14 = 14, Pyr21 = 125, Pyr29 = 126, Pyr30 = 118, Pyr37 = 127, Pyr45 = 128, Pyr53 = 129, Pyr55 = 119, Pyr61 = 130, Pyr69 = 131, Pyr91 = 120, Pyr140 = 121, Pyr204 = 122, Pyr285 = 123, Pyr385 = 124,
}

#[rustfmt::skip]
impl GmshElementType {
    pub fn n_nodes(&self) -> Result<usize> {
        match self {
            GmshElementType::Pnt => Ok(1),
            GmshElementType::Lin1 => Ok(1), GmshElementType::Lin2 => Ok(2), GmshElementType::Lin3 => Ok(3), GmshElementType::Lin4 => Ok(4), GmshElementType::Lin5 => Ok(5), GmshElementType::Lin6 => Ok(6), GmshElementType::Lin7 => Ok(7), GmshElementType::Lin8 => Ok(8), GmshElementType::Lin9 => Ok(9), GmshElementType::Lin10 => Ok(10), GmshElementType::Lin11 => Ok(11),
            GmshElementType::Tri1 => Ok(1), GmshElementType::Tri3 => Ok(3), GmshElementType::Tri6 => Ok(6), GmshElementType::Tri9 => Ok(9), GmshElementType::Tri10 => Ok(10), GmshElementType::Tri12 => Ok(12), GmshElementType::Tri15 => Ok(15), GmshElementType::Tri15i => Ok(15), GmshElementType::Tri18 => Ok(18), GmshElementType::Tri21 => Ok(21), GmshElementType::Tri21i => Ok(21), GmshElementType::Tri24 => Ok(24), GmshElementType::Tri27 => Ok(27), GmshElementType::Tri28 => Ok(28), GmshElementType::Tri30 => Ok(30), GmshElementType::Tri36 => Ok(36), GmshElementType::Tri45 => Ok(45), GmshElementType::Tri55 => Ok(55), GmshElementType::Tri66 => Ok(66), 
            GmshElementType::Qua1 => Ok(1), GmshElementType::Qua4 => Ok(4), GmshElementType::Qua8 => Ok(8), GmshElementType::Qua9 => Ok(9), GmshElementType::Qua12 => Ok(12), GmshElementType::Qua16 => Ok(16), GmshElementType::Qua16i => Ok(16), GmshElementType::Qua20 => Ok(20), GmshElementType::Qua24 => Ok(24), GmshElementType::Qua25 => Ok(25), GmshElementType::Qua28 => Ok(28), GmshElementType::Qua32 => Ok(32), GmshElementType::Qua36 => Ok(36), GmshElementType::Qua36i => Ok(36), GmshElementType::Qua40 => Ok(40), GmshElementType::Qua49 => Ok(49), GmshElementType::Qua64 => Ok(64), GmshElementType::Qua81 => Ok(81), GmshElementType::Qua100 => Ok(100), GmshElementType::Qua121 => Ok(121),
            GmshElementType::Tet1 => Ok(1), GmshElementType::Tet4 => Ok(4), GmshElementType::Tet10 => Ok(10), GmshElementType::Tet16 => Ok(16), GmshElementType::Tet20 => Ok(20), GmshElementType::Tet22 => Ok(22), GmshElementType::Tet28 => Ok(28), GmshElementType::Tet34 => Ok(34), GmshElementType::Tet35 => Ok(35), GmshElementType::Tet40 => Ok(40), GmshElementType::Tet46 => Ok(46), GmshElementType::Tet52 => Ok(52), GmshElementType::Tet56 => Ok(56), GmshElementType::Tet58 => Ok(58), GmshElementType::Tet84 => Ok(84), GmshElementType::Tet120 => Ok(120), GmshElementType::Tet165 => Ok(165), GmshElementType::Tet220 => Ok(220), GmshElementType::Tet286 => Ok(286),
            GmshElementType::Hex1 => Ok(1), GmshElementType::Hex8 => Ok(8), GmshElementType::Hex20 => Ok(20), GmshElementType::Hex27 => Ok(27), GmshElementType::Hex32 => Ok(32), GmshElementType::Hex44 => Ok(44), GmshElementType::Hex56 => Ok(56), GmshElementType::Hex64 => Ok(64), GmshElementType::Hex68 => Ok(68), GmshElementType::Hex80 => Ok(80), GmshElementType::Hex92 => Ok(92), GmshElementType::Hex104 => Ok(104), GmshElementType::Hex125 => Ok(125), GmshElementType::Hex216 => Ok(216), GmshElementType::Hex343 => Ok(343), GmshElementType::Hex512 => Ok(512), GmshElementType::Hex729 => Ok(729), GmshElementType::Hex1000 => Ok(1000),
            GmshElementType::Pri1 => Ok(1), GmshElementType::Pri6 => Ok(6), GmshElementType::Pri15 => Ok(15), GmshElementType::Pri18 => Ok(28), GmshElementType::Pri24 => Ok(24), GmshElementType::Pri33 => Ok(33), GmshElementType::Pri40 => Ok(40), GmshElementType::Pri42 => Ok(42), GmshElementType::Pri51 => Ok(51), GmshElementType::Pri60 => Ok(60), GmshElementType::Pri69 => Ok(69), GmshElementType::Pri75 => Ok(75), GmshElementType::Pri78 => Ok(78), GmshElementType::Pri126 => Ok(126), GmshElementType::Pri196 => Ok(196), GmshElementType::Pri288 => Ok(288), GmshElementType::Pri405 => Ok(405), GmshElementType::Pri550 => Ok(550),
            GmshElementType::Pyr1 => Ok(1), GmshElementType::Pyr5 => Ok(5), GmshElementType::Pyr13 => Ok(13), GmshElementType::Pyr14 => Ok(14), GmshElementType::Pyr21 => Ok(21), GmshElementType::Pyr29 => Ok(29), GmshElementType::Pyr30 => Ok(30), GmshElementType::Pyr37 => Ok(37), GmshElementType::Pyr45 => Ok(45), GmshElementType::Pyr53 => Ok(53), GmshElementType::Pyr55 => Ok(55), GmshElementType::Pyr61 => Ok(61), GmshElementType::Pyr69 => Ok(69), GmshElementType::Pyr91 => Ok(91), GmshElementType::Pyr140 => Ok(140), GmshElementType::Pyr204 => Ok(204), GmshElementType::Pyr285 => Ok(285), GmshElementType::Pyr385 => Ok(385),
            _ => bail!("{:?} does not have a constant number of nodes", self)
        }
    }
}
