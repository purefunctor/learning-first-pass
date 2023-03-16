use std::sync::Arc;

use crate::{
    material::Scatter,
    ray::Ray,
    vec3::{Point3, Vec3},
};

pub struct Hit {
    pub point: Point3,
    pub normal: Vec3,
    pub material: Arc<dyn Scatter>,
    pub t: f64,
    pub front_face: bool,
}

impl Hit {
    pub fn set_face_normal(&mut self, r: &Ray, outward_normal: Vec3) {
        self.front_face = r.direction.dot(outward_normal) < 0.0;
        self.normal = if self.front_face {
            outward_normal
        } else {
            -1.0 * outward_normal
        }
    }
}

pub trait Hittable: Send + Sync {
    fn hit(&self, r: &Ray, t_min: f64, t_max: f64) -> Option<Hit>;
}

pub type World = Vec<Box<dyn Hittable>>;

impl Hittable for World {
    fn hit(&self, r: &Ray, t_min: f64, t_max: f64) -> Option<Hit> {
        let mut current_hit = None;
        let mut closest_so_far = t_max;

        for object in self {
            if let Some(hit) = object.hit(r, t_min, closest_so_far) {
                closest_so_far = hit.t;
                current_hit = Some(hit);
            }
        }

        current_hit
    }
}
