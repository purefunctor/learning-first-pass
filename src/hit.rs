use crate::{
    ray::Ray,
    vec3::{Point3, Vec3},
    world::Object,
};

pub struct Hit {
    pub point: Point3,
    pub normal: Vec3,
    pub object: Object,
    pub t: f64,
    pub front_face: bool,
}

impl Hit {
    pub fn set_face_normal(&mut self, ray: &Ray, outward_normal: Vec3) {
        self.front_face = ray.direction.dot(outward_normal) < 0.0;
        self.normal = if self.front_face {
            outward_normal
        } else {
            -1.0 * outward_normal
        }
    }

    pub fn origin_features(&self) -> [f64; 3] {
        [self.point.x(), self.point.y(), self.point.z()]
    }

    pub fn normal_features(&self) -> [f64; 3] {
        [self.point.x(), self.point.y(), self.point.z()]
    }
}
