use std::{ops::Range, sync::Mutex};

use once_cell::sync::Lazy;
use rand::{rngs::StdRng, Rng, SeedableRng};

pub static GLOBAL_SEED: Lazy<Seed> = Lazy::new(|| Seed {
    inner: Mutex::new(StdRng::seed_from_u64(0)),
});

pub struct Seed {
    inner: Mutex<StdRng>,
}

impl Seed {
    pub fn set_seed(seed: u64) {
        *GLOBAL_SEED.inner.lock().unwrap() = StdRng::seed_from_u64(seed);
    }

    pub fn gen() -> f64 {
        GLOBAL_SEED.inner.lock().unwrap().gen()
    }

    pub fn gen_range(r: Range<f64>) -> f64 {
        GLOBAL_SEED.inner.lock().unwrap().gen_range(r)
    }
}
