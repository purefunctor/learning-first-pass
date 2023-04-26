use std::ops::{Add, Mul, AddAssign};

use itertools::iproduct;
use ndarray::Array3;
use numpy::ToPyArray;
use pyo3::{
    pyfunction, pymodule,
    types::{PyModule, PyTuple},
    wrap_pyfunction, PyResult, Python,
};
use rayon::prelude::{ParallelBridge, ParallelIterator};
use vecmath::{vec3_add, vec3_dot, vec3_mul, vec3_neg, vec3_normalized, vec3_sub, Vector3};

#[derive(Clone, Copy)]
pub struct Color {
    pub red: f64,
    pub green: f64,
    pub blue: f64,
}

impl Color {
    pub fn clamp(&self) -> Color {
        Color {
            red: self.red.clamp(0.0, 1.0),
            green: self.green.clamp(0.0, 1.0),
            blue: self.blue.clamp(0.0, 1.0),
        }
    }
}

impl Add for Color {
    type Output = Color;

    fn add(self, other: Self) -> Self::Output {
        Color {
            red: self.red + other.red,
            green: self.green + other.green,
            blue: self.blue + other.blue,
        }
    }
}

impl AddAssign for Color {
    fn add_assign(&mut self, other: Self) {
        self.red += other.red;
        self.green += other.green;
        self.blue += other.blue;
    }
}

impl Mul<f64> for Color {
    type Output = Color;

    fn mul(self, other: f64) -> Self::Output {
        Color {
            red: self.red * other,
            green: self.green * other,
            blue: self.blue * other,
        }
    }
}

impl Mul<Color> for Color {
    type Output = Color;

    fn mul(self, other: Color) -> Self::Output {
        Color {
            red: self.red * other.red,
            green: self.green * other.green,
            blue: self.blue * other.blue,
        }
    }
}

pub struct Sphere {
    pub origin: Vector3<f64>,
    pub radius: f64,
    pub color: Color,
    pub albedo: f64,
}

pub struct Plane {
    pub origin: Vector3<f64>,
    pub normal: Vector3<f64>,
    pub color: Color,
    pub albedo: f64,
}

pub enum Light {
    Directional {
        direction: Vector3<f64>,
        color: Color,
        intensity: f64,
    },
}
impl Light {
    fn direction(&self, _: &Vector3<f64>) -> Vector3<f64> {
        match self {
            Light::Directional { direction, .. } => vec3_neg(vec3_normalized(*direction)),
        }
    }

    fn intensity(&self, _: &Vector3<f64>) -> f64 {
        match self {
            Light::Directional { intensity, .. } => *intensity,
        }
    }

    fn color(&self) -> &Color {
        match self {
            Light::Directional { color, .. } => color,
        }
    }

    fn distance(&self, _: &Vector3<f64>) -> f64 {
        match self {
            Light::Directional { .. } => std::f64::INFINITY,
        }
    }
}

pub enum Element {
    Sphere(Sphere),
    Plane(Plane),
}

impl Element {
    pub fn color(&self) -> &Color {
        match self {
            Element::Sphere(s) => &s.color,
            Element::Plane(p) => &p.color,
        }
    }

    pub fn albedo(&self) -> f64 {
        match self {
            Element::Sphere(s) => s.albedo,
            Element::Plane(p) => p.albedo,
        }
    }
}

pub struct Scene {
    pub width: usize,
    pub height: usize,
    pub fov: f64,
    pub elements: Vec<Element>,
    pub lights: Vec<Light>,
    pub shadow_bias: f64,
}

pub struct Intersection<'a> {
    pub distance: f64,
    pub element: &'a Element,
}

impl<'a> Intersection<'a> {
    pub fn new(distance: f64, element: &'a Element) -> Intersection<'a> {
        Intersection { distance, element }
    }
}

impl Scene {
    pub fn trace(&self, ray: &Ray) -> Option<Intersection> {
        self.elements
            .iter()
            .filter_map(|s| s.intersect(ray).map(|d| Intersection::new(d, s)))
            .min_by(|i1, i2| i1.distance.partial_cmp(&i2.distance).unwrap())
    }
}

pub struct Ray {
    pub origin: Vector3<f64>,
    pub direction: Vector3<f64>,
}

impl Ray {
    pub fn create_prime(x: usize, y: usize, scene: &Scene) -> Ray {
        let fov_adjustment = (scene.fov.to_radians() / 2.0).tan();
        let aspect_ratio = (scene.width as f64) / (scene.height as f64);
        let sensor_x =
            ((((x as f64 + 0.5) / scene.width as f64) * 2.0 - 1.0) * aspect_ratio) * fov_adjustment;
        let sensor_y = (1.0 - ((y as f64 + 0.5) / scene.height as f64) * 2.0) * fov_adjustment;

        Ray {
            origin: [0.0, 0.0, 0.0],
            direction: vec3_normalized([sensor_x, sensor_y, -1.0]),
        }
    }
}

pub trait Intersectable {
    fn intersect(&self, ray: &Ray) -> Option<f64>;

    fn surface_normal(&self, hit_point: &Vector3<f64>) -> Vector3<f64>;
}

impl Intersectable for Sphere {
    fn intersect(&self, ray: &Ray) -> Option<f64> {
        let hypotenuse = vec3_sub(self.origin, ray.origin);
        let adjacent = vec3_dot(hypotenuse, ray.direction);
        let opposite = vec3_dot(hypotenuse, hypotenuse) - (adjacent * adjacent);
        let radius_s = self.radius * self.radius;

        if opposite > self.radius {
            return None;
        }

        let adjacent_inside = (radius_s - opposite).sqrt();
        let adjacent_left_half = adjacent - adjacent_inside;
        let adjacent_right_half = adjacent + adjacent_inside;

        if adjacent_left_half < 0.0 && adjacent_right_half < 0.0 {
            return None;
        }

        Some(adjacent_left_half.min(adjacent_right_half))
    }

    fn surface_normal(&self, hit_point: &Vector3<f64>) -> Vector3<f64> {
        vec3_normalized(vec3_sub(*hit_point, self.origin))
    }
}

impl Intersectable for Plane {
    fn intersect(&self, ray: &Ray) -> Option<f64> {
        let denominator = vec3_dot(self.normal, ray.direction);
        if denominator > 1e-6 {
            let v = vec3_sub(self.origin, ray.origin);
            let distance = vec3_dot(v, self.normal) / denominator;
            if distance >= 0.0 {
                return Some(distance);
            }
        }
        None
    }

    fn surface_normal(&self, _: &Vector3<f64>) -> Vector3<f64> {
        vec3_neg(self.normal)
    }
}

impl Intersectable for Element {
    fn intersect(&self, ray: &Ray) -> Option<f64> {
        match self {
            Element::Sphere(s) => s.intersect(ray),
            Element::Plane(p) => p.intersect(ray),
        }
    }

    fn surface_normal(&self, hit_point: &Vector3<f64>) -> Vector3<f64> {
        match self {
            Element::Sphere(s) => s.surface_normal(hit_point),
            Element::Plane(p) => p.surface_normal(hit_point),
        }
    }
}

fn get_color(scene: &Scene, ray: &Ray, intersection: &Intersection) -> Color {
    let hit_point = vec3_add(
        ray.origin,
        vec3_mul(ray.direction, [intersection.distance; 3]),
    );
    let surface_normal = intersection.element.surface_normal(&hit_point);

    let mut color = Color {
        red: 0.0,
        green: 0.0,
        blue: 0.0,
    };

    for light in &scene.lights {
        let direction_to_light = light.direction(&hit_point);

        let shadow_ray = Ray {
            origin: vec3_add(hit_point, vec3_mul(surface_normal, [scene.shadow_bias; 3])),
            direction: direction_to_light,
        };
        let shadow_intersection = scene.trace(&shadow_ray);
        let in_light = if let Some(intersection) = shadow_intersection {
            intersection.distance > light.distance(&hit_point)
        } else {
            true
        };

        let light_intensity = if in_light {
            light.intensity(&hit_point)
        } else {
            0.0
        };
        let light_power = vec3_dot(surface_normal, direction_to_light).max(0.0) * light_intensity;
        let light_reflected = intersection.element.albedo() / std::f64::consts::PI;

        let element_color = *intersection.element.color();
        let light_color = *light.color() * light_power * light_reflected;

        color += element_color * light_color;
    }

    color.clamp()
}

const CHANNELS: usize = 3;

#[pyfunction]
fn render(py: Python<'_>, width: usize, height: usize, fov: f64) -> PyResult<&'_ PyTuple> {
    let mut image_buffer: Array3<f64> = Array3::zeros((height, width, CHANNELS));

    let scene = Scene {
        width,
        height,
        fov,
        elements: vec![
            Element::Sphere(Sphere {
                origin: [0.0, 0.0, -5.0],
                radius: 1.0,
                color: Color {
                    red: 0.4,
                    green: 1.0,
                    blue: 0.4,
                },
                albedo: 0.20,
            }),
            Element::Sphere(Sphere {
                origin: [1.0, 1.0, -4.0],
                radius: 1.5,
                color: Color {
                    red: 0.7,
                    green: 0.5,
                    blue: 0.3,
                },
                albedo: 0.20,
            }),
            Element::Plane(Plane {
                origin: [0.0, -2.0, -5.0],
                normal: [0.0, -1.0, 0.0],
                color: Color {
                    red: 0.5,
                    blue: 0.5,
                    green: 0.5,
                },
                albedo: 0.20,
            }),
        ],
        lights: vec![Light::Directional {
            direction: [-0.5, -1.0, -1.0],
            color: Color {
                red: 1.0,
                blue: 1.0,
                green: 1.0,
            },
            intensity: 20.0,
        }],
        shadow_bias: 1e-13,
    };

    let pixels = iproduct!(0..width, 0..height)
        .par_bridge()
        .map(|(x, y)| {
            let ray = Ray::create_prime(x, y, &scene);

            if let Some(intersection) = scene.trace(&ray) {
                let Color { red, green, blue } = get_color(&scene, &ray, &intersection);
                (x, y, red, green, blue)
            } else {
                let unit_direction = vec3_normalized(ray.direction);
                let t = 0.5 * (unit_direction[1] + 1.0);
                let r = (1.0 - t) * 1.0 + t * 0.5;
                let g = (1.0 - t) * 1.0 + t * 0.7;
                let b = (1.0 - t) * 1.0 + t * 1.0;
                (x, y, r, g, b)
            }
        })
        .collect::<Vec<_>>();

    for (x, y, r, g, b) in pixels {
        image_buffer[[y, x, 0]] = r;
        image_buffer[[y, x, 1]] = g;
        image_buffer[[y, x, 2]] = b;
    }

    Ok(PyTuple::new(py, &[image_buffer.to_pyarray(py)]))
}

#[pymodule]
fn cpu_tracer(_: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(render, m)?)?;
    Ok(())
}
