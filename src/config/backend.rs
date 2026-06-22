use serde::Deserialize;

#[derive(Deserialize, Debug, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum BackendPrecision {
    Single,
    Double,
}

#[derive(Deserialize, Debug)]
pub struct BackendConfig {
    precision: BackendPrecision,
}
