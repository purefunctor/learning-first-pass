use std::{
    fmt::Display,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Range, Sub, SubAssign},
};

use rand::Rng;
use rand_chacha::ChaCha8Rng;

#[derive(Clone, Copy, Debug)]
pub struct Vec3 {
    e: [f64; 3],
}

pub type Point3 = Vec3;

pub type Color = Vec3;

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { e: [x, y, z] }
    }

    pub fn x(self) -> f64 {
        self[0]
    }

    pub fn y(self) -> f64 {
        self[1]
    }

    pub fn z(self) -> f64 {
        self[2]
    }

    pub fn as_tuple(self) -> (f64, f64, f64) {
        (self[0], self[1], self[2])
    }

    pub fn dot(self, other: Vec3) -> f64 {
        self[0] * other[0] + self[1] * other[1] + self[2] * other[2]
    }

    pub fn length(self) -> f64 {
        self.dot(self).sqrt()
    }

    pub fn length_squared(self) -> f64 {
        self.dot(self)
    }

    pub fn cross(self, other: Vec3) -> Vec3 {
        Vec3 {
            e: [
                self[1] * other[2] - self[2] * other[1],
                self[2] * other[0] - self[0] * other[2],
                self[0] * other[1] - self[1] * other[0],
            ],
        }
    }

    pub fn normalized(self) -> Vec3 {
        self / self.length()
    }

    pub fn format_color(self, samples_per_pixel: usize) -> String {
        let ir: usize = (256.0
            * (self[0] / (samples_per_pixel as f64))
                .sqrt()
                .clamp(0.0, 0.999)) as usize;
        let ig: usize = (256.0
            * (self[1] / (samples_per_pixel as f64))
                .sqrt()
                .clamp(0.0, 0.999)) as usize;
        let ib: usize = (256.0
            * (self[2] / (samples_per_pixel as f64))
                .sqrt()
                .clamp(0.0, 0.999)) as usize;

        format!("{} {} {}", ir, ig, ib)
    }

    pub fn random(rng: &mut ChaCha8Rng, r: Range<f64>) -> Self {
        Self {
            e: [
                rng.gen_range(r.clone()),
                rng.gen_range(r.clone()),
                rng.gen_range(r),
            ],
        }
    }

    pub fn random_in_unit_sphere(rng: &mut ChaCha8Rng) -> Self {
        loop {
            let v = Vec3::random(rng, -1.0..1.0);
            if v.length_squared() < 1.0 {
                return v;
            }
        }
    }

    pub fn random_in_hemisphere(rng: &mut ChaCha8Rng, normal: Vec3) -> Vec3 {
        let in_unit_sphere = Self::random_in_unit_sphere(rng);
        if in_unit_sphere.dot(normal) > 0.0 {
            in_unit_sphere
        } else {
            -1.0 * in_unit_sphere
        }
    }

    pub fn near_zero(self) -> bool {
        const EPS: f64 = 1.0e-8;
        self[0].abs() < EPS && self[1].abs() < EPS && self[2].abs() < EPS
    }

    pub fn reflect(self, n: Vec3) -> Vec3 {
        self - 2.0 * self.dot(n) * n
    }

    pub fn refract(self, n: Vec3, etai_over_etat: f64) -> Vec3 {
        let cos_theta = (-1.0 * self).dot(n).min(1.0);
        let r_out_perp = etai_over_etat * (self + cos_theta * n);
        let r_out_parallel = -(1.0 - r_out_perp.length().powi(2)).abs().sqrt() * n;
        r_out_perp + r_out_parallel
    }

    pub fn random_in_unit_disk(rng: &mut ChaCha8Rng) -> Vec3 {
        loop {
            let p = Vec3::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0), 0.0);
            if p.length_squared() < 1.0 {
                return p;
            }
        }
    }
}

impl Index<usize> for Vec3 {
    type Output = f64;
    fn index(&self, index: usize) -> &f64 {
        &self.e[index]
    }
}

impl IndexMut<usize> for Vec3 {
    fn index_mut(&mut self, index: usize) -> &mut f64 {
        &mut self.e[index]
    }
}

impl Add for Vec3 {
    type Output = Vec3;

    fn add(self, other: Vec3) -> Vec3 {
        Vec3 {
            e: [self[0] + other[0], self[1] + other[1], self[2] + other[2]],
        }
    }
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, other: Vec3) {
        *self = Vec3 {
            e: [self[0] + other[0], self[1] + other[1], self[2] + other[2]],
        }
    }
}

impl Sub for Vec3 {
    type Output = Vec3;

    fn sub(self, other: Vec3) -> Vec3 {
        Vec3 {
            e: [self[0] - other[0], self[1] - other[1], self[2] - other[2]],
        }
    }
}

impl SubAssign for Vec3 {
    fn sub_assign(&mut self, other: Vec3) -> () {
        *self = Vec3 {
            e: [self[0] - other[0], self[1] - other[1], self[2] - other[2]],
        };
    }
}

impl Mul<f64> for Vec3 {
    type Output = Vec3;

    fn mul(self, other: f64) -> Vec3 {
        Vec3 {
            e: [self[0] * other, self[1] * other, self[2] * other],
        }
    }
}

impl MulAssign<f64> for Vec3 {
    fn mul_assign(&mut self, other: f64) -> () {
        *self = Vec3 {
            e: [self[0] * other, self[1] * other, self[2] * other],
        };
    }
}

impl MulAssign<Vec3> for Vec3 {
    fn mul_assign(&mut self, other: Vec3) {
        self[0] = self[0] * other[0];
        self[1] = self[1] * other[1];
        self[2] = self[2] * other[2];
    }
}

impl Mul<Vec3> for f64 {
    type Output = Vec3;

    fn mul(self, other: Vec3) -> Vec3 {
        Vec3 {
            e: [self * other[0], self * other[1], self * other[2]],
        }
    }
}

impl Mul<Vec3> for Vec3 {
    type Output = Vec3;

    fn mul(self, other: Vec3) -> Vec3 {
        Vec3 {
            e: [self[0] * other[0], self[1] * other[1], self[2] * other[2]],
        }
    }
}

impl Div<f64> for Vec3 {
    type Output = Vec3;

    fn div(self, other: f64) -> Vec3 {
        Vec3 {
            e: [self[0] / other, self[1] / other, self[2] / other],
        }
    }
}

impl DivAssign<f64> for Vec3 {
    fn div_assign(&mut self, other: f64) -> () {
        *self = Vec3 {
            e: [self[0] / other, self[1] / other, self[2] / other],
        };
    }
}

impl Display for Vec3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {})", self[0], self[1], self[2])
    }
}
