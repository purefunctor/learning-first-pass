use itertools::iproduct;
use ndarray::Array3;
use numpy::ToPyArray;
use pyo3::{
    pyfunction, pymodule,
    types::{PyModule, PyTuple},
    wrap_pyfunction, PyResult, Python,
};
use rayon::prelude::{ParallelBridge, ParallelIterator};
use vecmath::{vec3_normalized, Vector3, vec3_sub, vec3_dot};

pub struct Color {
    pub red: f64,
    pub blue: f64,
    pub green: f64,
}

pub struct Sphere {
    pub origin: Vector3<f64>,
    pub radius: f64,
    pub color: Color,
}

pub struct Scene {
    pub width: usize,
    pub height: usize,
    pub fov: f64,
    pub sphere: Sphere,
}

pub struct Ray {
    pub origin: Vector3<f64>,
    pub direction: Vector3<f64>,
}

impl Ray {
    pub fn create_prime(x: usize, y: usize, scene: &Scene) -> Ray {
        let fov_adjustment = (scene.fov.to_radians() / 2.0).tan();
        let aspect_ratio = (scene.width as f64) / (scene.height as f64);
        let sensor_x = ((((x as f64 + 0.5) / scene.width as f64) * 2.0 - 1.0) * aspect_ratio) * fov_adjustment;
        let sensor_y = (1.0 - ((y as f64 + 0.5) / scene.height as f64) * 2.0) * fov_adjustment;

        Ray {
            origin: [0.0, 0.0, 0.0],
            direction: vec3_normalized([sensor_x, sensor_y, -1.0]),
        }
    }
}

pub trait Intersectable {
    fn intersect(&self, ray: &Ray) -> Option<f64>;
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

    let thc = (radius_s - opposite).sqrt();
       let t0 = adjacent - thc;
       let t1 = adjacent + thc;

       if t0 < 0.0 && t1 < 0.0 {
           return None;
       }

       let distance = if t0 < t1 { t0 } else { t1 };
       Some(distance)
    }
}

const CHANNELS: usize = 3;

#[pyfunction]
fn render(py: Python<'_>, width: usize, height: usize, fov: f64) -> PyResult<&'_ PyTuple> {
    let mut image_buffer: Array3<f64> = Array3::zeros((height, width, CHANNELS));

    let scene = Scene {
        width,
        height,
        fov,
        sphere: Sphere {
            origin: [0.0, 0.0, -5.0],
            radius: 1.0,
            color: Color {
                red: 0.4,
                green: 1.0,
                blue: 0.4,
            },
        },
    };

    let pixels = iproduct!(0..width, 0..height)
        .par_bridge()
        .map(|(x, y)| {
            let ray = Ray::create_prime(x, y, &scene);

            if let Some(_) = scene.sphere.intersect(&ray) {
                (x, y, scene.sphere.color.red, scene.sphere.color.green, scene.sphere.color.blue)
            } else {
                (x, y, 0.0, 0.0, 0.0)   
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
