//! Regressions against the actual patched truck constructor, run by workspace CI.
use truck_meshalgo::prelude::PolylineCurve;
use truck_modeling::{BoundedCurve, ParametricCurve, Plane, Point3, Sphere};
use truck_shapeops::IntersectionCurveWithParameters;

fn planes() -> (Plane, Plane) {
    let o = Point3::new(0.0, 0.0, 0.0);
    (Plane::new(o, Point3::new(1.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)), Plane::new(o, Point3::new(1.0, 0.0, 0.0), Point3::new(0.0, 0.0, 1.0)))
}

#[test]
fn degenerate_intersection_polylines_return_none_without_panicking() {
    let origin = Point3::new(0.0, 0.0, 0.0);
    for points in [
        vec![],
        vec![origin],
        vec![origin, origin],
        vec![origin; 3],
        vec![origin, Point3::new(f64::NAN, 0.0, 0.0)],
        vec![origin, Point3::new(f64::INFINITY, 0.0, 0.0)],
    ] {
        let (a, b) = planes();
        let result = std::panic::catch_unwind(|| IntersectionCurveWithParameters::try_new(a, b, PolylineCurve(points.clone()), 0.01));
        assert!(result.is_ok(), "constructor panicked for {points:?}");
        assert!(result.unwrap().is_none(), "collapsed input must not create a curve: {points:?}");
    }
}

#[test]
fn closed_intersection_retains_the_closure_and_spherical_supports() {
    let a = Sphere::new(Point3::new(0.0, 0.0, 1.0), 2.0_f64.sqrt());
    let b = Sphere::new(Point3::new(0.0, 0.0, -1.0), 2.0_f64.sqrt());
    let poly = PolylineCurve(vec![
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
        Point3::new(-1.0, 0.0, 0.0),
        Point3::new(0.0, -1.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
    ]);
    let curve = IntersectionCurveWithParameters::try_new(a, b, poly, 0.01).unwrap();
    let (lo, hi) = curve.range_tuple();
    assert_eq!(curve.subs(lo), curve.subs(hi));
    for i in 0..=40 {
        let p = curve.subs(lo + (hi - lo) * i as f64 / 40.0);
        assert!(p.z.abs() < 1e-8, "{p:?}");
        assert!(((p.x * p.x + p.y * p.y).sqrt() - 1.0).abs() <= 0.01, "{p:?}");
    }
}

#[test]
fn open_intersection_keeps_endpoints_on_both_planes() {
    let (a, b) = planes();
    let start = Point3::new(1.0, 0.0, 0.0);
    let end = Point3::new(3.0, 0.0, 0.0);
    let curve = IntersectionCurveWithParameters::try_new(a, b, PolylineCurve(vec![start, end]), 0.01).unwrap();
    let (lo, hi) = curve.range_tuple();
    assert_eq!(curve.subs(lo), start);
    assert_eq!(curve.subs(hi), end);
    assert_eq!(curve.subs((lo + hi) / 2.0), Point3::new(2.0, 0.0, 0.0));
}

/// A triangular face whose first edge is an intersection curve of two planes: the parallel planes
/// `z = 0` and `z = 1` when `support_gap` is `1.0` (the projection onto both never converges), the
/// same plane twice when it is `0.0` (the leader lies on both and always evaluates).
fn triangle_with_intersection_edge(support_gap: f64) -> truck_modeling::Shell {
    use truck_modeling::{Curve, Edge, Face, IntersectionCurve, Leader, Line, Shell, Surface, Vertex, Wire};
    let plane = |z: f64| Surface::Plane(Plane::new(Point3::new(0.0, 0.0, z), Point3::new(1.0, 0.0, z), Point3::new(0.0, 1.0, z)));
    let (v0, v1, v2) = (Vertex::new(Point3::new(0.0, 0.0, 0.5)), Vertex::new(Point3::new(1.0, 0.0, 0.5)), Vertex::new(Point3::new(0.5, 1.0, 0.5)));
    let leader = Leader::Polyline(PolylineCurve(vec![Point3::new(0.0, 0.0, 0.5), Point3::new(1.0, 0.0, 0.5)]));
    let bad = IntersectionCurve::new_unchecked(Box::new(plane(0.5 - support_gap / 2.0)), Box::new(plane(0.5 + support_gap / 2.0)), leader, 0.01);
    let line = |a: &Vertex, b: &Vertex| Edge::new(a, b, Curve::Line(Line(a.point(), b.point())));
    let wire = Wire::from(vec![Edge::new(&v0, &v1, Curve::IntersectionCurve(bad)), line(&v1, &v2), line(&v2, &v0)]);
    Shell::from(vec![Face::new(vec![wire], plane(0.5))])
}

/// Meshing an edge whose intersection curve does not evaluate (#1252): `IntersectionCurve::subs`
/// unwraps a failed projection, and the mesher sampled the edge through it. The two supports here
/// are parallel planes, so the projection never converges. Nothing may panic (wasm aborts on one,
/// and the native guard prints the message first) and the mesh stays finite.
#[test]
fn meshing_an_edge_that_does_not_evaluate_neither_panics_nor_goes_non_finite() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use truck_meshalgo::prelude::RobustMeshableShape;
    let shell = triangle_with_intersection_edge(1.0);

    let me = std::thread::current().id();
    let count = Arc::new(AtomicUsize::new(0));
    let c2 = count.clone();
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |_| {
        if std::thread::current().id() == me {
            c2.fetch_add(1, Ordering::SeqCst);
        }
    }));
    let meshed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| shell.robust_triangulation(0.01)));
    std::panic::set_hook(prev);
    assert_eq!(count.load(Ordering::SeqCst), 0, "panics while meshing");
    let meshed = meshed.unwrap();
    for face in meshed.face_iter() {
        if let Some(mesh) = face.surface() {
            assert!(mesh.positions().iter().all(|p| p.x.is_finite() && p.y.is_finite() && p.z.is_finite()));
        }
    }
}

/// A boolean's result with an edge that does not evaluate is refused with an error (#1252). The
/// mesher's panic on such an edge used to reject the attempt; now that it no longer panics, the
/// result must be refused explicitly or a plausible-looking mesh would let it through.
#[test]
fn a_result_with_an_edge_that_does_not_evaluate_is_refused() {
    use truck_modeling::Solid;
    let solid = |gap: f64| Solid::new_unchecked(vec![triangle_with_intersection_edge(gap)]);
    assert!(!crate::boolean_record::edges_evaluate(&solid(1.0)));
    assert!(crate::boolean_record::edges_evaluate(&solid(0.0)));
}

/// A circular pattern of lugs round a bore whose inner walls are the bore's own wall, after a
/// fillet on the first lug turned the base's faces into free-form surfaces: each copy's wall
/// then meets the bore's in an intersection of two surfaces that all but coincide. The division
/// of those curves never settled, so the pattern ran for minutes at the reported bore (one join
/// 90 s) and did not finish at others, and in the application the recompute never seemed to
/// finish. Alone each takes about 6 s; the limit leaves room for a run of the whole suite.
mod lugs_round_a_filleted_bore {
    use std::sync::mpsc;
    use std::time::Duration;

    use solidcraft_geom::{Loop2, Plane as SketchPlane, Region2, Seg2, Vec2, Vec3};

    use crate::{Body, BoolOp, boolean, extrude, fillet, transform_matrix};

    const LUG_OUTER_RADIUS: f64 = 10.75;
    const LUG_FIRST_DEGREES: f64 = 103.0;
    const LUG_SPAN_DEGREES: f64 = 30.0;
    const LUG_FOOT: f64 = 6.0;
    const LUG_TOP: f64 = 31.0;
    const LIMIT: Duration = Duration::from_secs(300);

    fn rotation_about_z(degrees: f64) -> [[f64; 4]; 4] {
        let (s, c) = degrees.to_radians().sin_cos();
        [[c, s, 0., 0.], [-s, c, 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]
    }

    /// The lug: a sector of the ring from the bore out to `LUG_OUTER_RADIUS`, standing on the base.
    fn lug(bore: f64) -> Body {
        let (start, sweep) = (LUG_FIRST_DEGREES.to_radians(), LUG_SPAN_DEGREES.to_radians());
        let at = |r: f64, a: f64| Vec2::new(r * a.cos(), r * a.sin());
        let origin = Vec2::new(0., 0.);
        let segs = vec![
            Seg2::Arc { center: origin, radius: LUG_OUTER_RADIUS, start, sweep },
            Seg2::Line { a: at(LUG_OUTER_RADIUS, start + sweep), b: at(bore, start + sweep) },
            Seg2::Arc { center: origin, radius: bore, start: start + sweep, sweep: -sweep },
            Seg2::Line { a: at(bore, start), b: at(LUG_OUTER_RADIUS, start) },
        ];
        extrude(&SketchPlane::XY, &[Region2 { outer: Loop2 { segs }, holes: vec![] }], LUG_FOOT, LUG_TOP).unwrap().remove(0)
    }

    /// A 64 mm disc, 6 mm thick, with the bore through it.
    fn base(bore: f64) -> Body {
        let region = Region2 { outer: Loop2::circle(Vec2::new(0., 0.), 32.), holes: vec![Loop2::circle(Vec2::new(0., 0.), bore).reversed()] };
        extrude(&SketchPlane::XY, &[region], 0., LUG_FOOT).unwrap().remove(0)
    }

    fn pattern_of_lugs(bore: f64, count: usize) -> Result<Body, String> {
        let seed = lug(bore);
        let joined = boolean(&base(bore), &seed, BoolOp::Union).map_err(|e| e.to_string())?.ok_or("the first lug joined to nothing")?;
        // The 2 mm fillet where the lug's outer wall meets the base.
        let middle = (LUG_FIRST_DEGREES + LUG_SPAN_DEGREES / 2.0).to_radians();
        let edge = Vec3::new(LUG_OUTER_RADIUS * middle.cos(), LUG_OUTER_RADIUS * middle.sin(), LUG_FOOT);
        let mut body = fillet(&joined, &[edge], 2.0).map_err(|e| e.to_string())?;
        for k in 1..count {
            let copy = transform_matrix(&seed, rotation_about_z(360.0 * k as f64 / count as f64)).map_err(|e| e.to_string())?;
            body = boolean(&body, &copy, BoolOp::Union).map_err(|e| e.to_string())?.ok_or(format!("copy {k} joined to nothing"))?;
        }
        Ok(body)
    }

    /// Runs the pattern on a thread of its own with a limit: a division that does not finish
    /// then fails the test where it would hang the run.
    fn within_the_limit(bore: f64, count: usize) -> Result<Body, String> {
        let (sender, receiver) = mpsc::channel();
        std::thread::Builder::new()
            .stack_size(64 << 20)
            .spawn(move || {
                let _ = sender.send(pattern_of_lugs(bore, count));
            })
            .unwrap();
        receiver.recv_timeout(LIMIT).expect("the pattern of lugs did not finish")
    }

    /// The bore of the design this was found in: it finished, after minutes, without the bounds.
    #[test]
    fn the_pattern_at_the_reported_bore_finishes_and_keeps_every_lug() {
        let bore = 6.0;
        let body = within_the_limit(bore, 5).expect("the pattern joins");
        let volume = body.tessellate_with(1e-3, true).unwrap().measure().volume;
        // The disc with its bore, five lugs, and the small fillet's material (under 10 mm³).
        let disc = std::f64::consts::PI * (32f64.powi(2) - bore.powi(2)) * LUG_FOOT;
        let lug = 0.5 * LUG_SPAN_DEGREES.to_radians() * (LUG_OUTER_RADIUS.powi(2) - bore.powi(2)) * (LUG_TOP - LUG_FOOT);
        let expected = disc + 5.0 * lug;
        assert!((volume - expected).abs() < 0.01 * expected, "volume {volume} for {expected} expected");
    }

    /// A bore at which the pattern did not finish at all without the bounds. Whether its joins
    /// can be computed reliably is not what is held here, only that the answer, a body or a
    /// refusal, comes.
    #[test]
    fn a_pattern_that_cannot_join_reliably_answers_instead_of_running_on() {
        let answer = within_the_limit(5.5, 5);
        if let Err(why) = answer {
            assert!(!why.is_empty());
        }
    }
}
