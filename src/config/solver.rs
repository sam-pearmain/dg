use std::collections::HashMap;

use serde::Deserialize;

use crate::config::solver::{elements::ElementsConfig, interfaces::InterfacesConfig};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct SolverConfig {
    #[serde(flatten)]
    system: SystemConfig,
    order: usize,
    time_integration: TimeIntegrationConfig,
    shock_capturing: Option<ShockCapturingConfig>,
    interfaces: InterfacesConfig,
    elements: ElementsConfig,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "kebab-case", tag = "system")]
pub enum SystemConfig {
    #[serde(rename_all = "kebab-case")]
    LinearAdvection {
        constants: LinearAdvectionConstants,
        boundary_conditions: HashMap<String, LinearAdvectionBoundaryCondition>,
    },
    #[serde(rename_all = "kebab-case")]
    Euler {
        constants: EulerConstants,
        boundary_conditions: HashMap<String, EulerBoundaryCondition>,
    },
    #[serde(rename_all = "kebab-case")]
    NavierStokes {
        constants: NavierStokesConstants,
        boundary_conditions: HashMap<String, NavierStokesBoundaryCondition>,
    },
}

#[derive(Debug, Deserialize)]
pub struct LinearAdvectionConstants {
    /// The wavespeed
    c: f64,
}

#[derive(Debug, Deserialize)]
pub struct EulerConstants {
    gamma: f64,
}

#[derive(Debug, Deserialize)]
pub struct NavierStokesConstants {
    gamma: f64,
    #[serde(rename = "Pr")]
    prandtl: f64,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "scheme", rename_all = "kebab-case")]
pub enum TimeIntegrationConfig {
    /// 1st-order explicit Euler
    #[serde(rename_all = "kebab-case")]
    Euler { t_start: f64, t_end: f64, dt: f64 },
    /// 3rd-order, four-stage embedded Runge-Kutta
    #[serde(rename_all = "kebab-case")]
    Rk34 { t_start: f64, t_end: f64, dt: f64 },
    /// 4th-order, standard Runge-Kutta
    #[serde(rename_all = "kebab-case")]
    Rk4 { t_start: f64, t_end: f64, dt: f64 },
    /// 4th-order, five-stage Runge-Kutta
    #[serde(rename_all = "kebab-case")]
    Rk45 { t_start: f64, t_end: f64, dt: f64 },
    /// 4th-order, total variation deminishing Runge-Kutta
    #[serde(rename_all = "kebab-case")]
    TvdRk3 { t_start: f64, t_end: f64, dt: f64 },
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "kebab-case", tag = "method")]
pub enum ShockCapturingConfig {
    #[serde(rename_all = "kebab-case")]
    ArtificialViscosity {
        max_artificial_viscosity: f64,
        s0: f64,
        kappa: f64,
    },
    #[serde(rename_all = "kebab-case")]
    EntropyFilter {
        rho_min: f64,
        p_min: f64,
        s_min: f64,
        num_iters: usize,
        formulation: EntropyFilterFormulation,
    },
}

#[derive(Debug, Deserialize)]
pub enum EntropyFilterFormulation {
    Nonlinear,
    Linearised,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum BoundaryConditionValue {
    Constant(f64),
    Expression(String),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum LinearAdvectionBoundaryCondition {
    Dirichlet { u: BoundaryConditionValue },
    Extrapolated,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum EulerBoundaryCondition {
    CharRiemInv {
        rho: BoundaryConditionValue,
        u: BoundaryConditionValue,
        v: Option<BoundaryConditionValue>,
        w: Option<BoundaryConditionValue>,
        p: BoundaryConditionValue,
    },
    SlipAdiabaticWall,
    // add more
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum NavierStokesBoundaryCondition {
    CharacteristicRiemannInvariance {
        rho: BoundaryConditionValue,
        u: BoundaryConditionValue,
        v: Option<BoundaryConditionValue>,
        w: Option<BoundaryConditionValue>,
        p: BoundaryConditionValue,
    },
    NoSlipAdiabaticWall,
    NoSlipIsothermalWall {
        u: BoundaryConditionValue,
        v: Option<BoundaryConditionValue>,
        w: Option<BoundaryConditionValue>,
        cptw: BoundaryConditionValue,
    },
    SlipAdiabaticWall,
    // add more
}

mod interfaces {
    use super::*;
    use crate::quadrules::quadrature::QuadratureType;

    #[derive(Debug, Deserialize)]
    #[serde(tag = "type", rename_all = "kebab-case")]
    pub enum RiemannSolverConfig {
        Rusanov(LocalDiscontinuousGalerkinConstants),
        Hll(LocalDiscontinuousGalerkinConstants),
        Hllc(LocalDiscontinuousGalerkinConstants),
        Roe(LocalDiscontinuousGalerkinConstants),
    }

    #[derive(Debug, Deserialize)]
    pub struct LocalDiscontinuousGalerkinConstants {
        beta: f64,
        tau: f64,
    }

    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "kebab-case")]
    pub struct InterfacesConfig {
        line: Option<InterfaceConfig>,
        tri: Option<InterfaceConfig>,
        quad: Option<InterfaceConfig>,
        riemann_solver: RiemannSolverConfig,
    }

    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "kebab-case")]
    struct InterfaceConfig {
        flux_points: QuadratureType,
        quadrature_degree: Option<usize>,
        quadrature_rule: Option<QuadratureType>,
    }
}

mod elements {
    use super::*;
    use crate::quadrules::quadrature::QuadratureType;

    #[derive(Debug, Deserialize)]
    pub struct ElementsConfig {
        line: Option<ElementConfig>,
        tri: Option<ElementConfig>,
        quad: Option<ElementConfig>,
        tet: Option<ElementConfig>,
        hex: Option<ElementConfig>,
        pri: Option<ElementConfig>,
        pyr: Option<ElementConfig>,
    }

    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "kebab-case")]
    struct ElementConfig {
        solution_points: QuadratureType,
        quadrature_degree: Option<usize>,
        quadrature_rule: Option<QuadratureType>,
    }
}
