//! Per-system timing telemetry.
//!
//! Demo systems wrap their body with [`record`], accumulating nanoseconds
//! per label into a global registry. Recording happens on worker threads and
//! costs one `Instant::now()` pair per system per tick and the runner drains 
//! the registry once per redraw to feed the HUD. This is deliberately 
//! coarse (per system, not per query):
//! its job is to answer "which system owns the SIM milliseconds".

use std::collections::HashMap;
use std::sync::Mutex;

/// Accumulated timing for one labeled system since the last drain.
#[derive(Debug, Clone, Copy, Default)]
pub struct Sample {
    /// Total nanoseconds recorded for the label.
    pub nanos: u64,
    /// How many times [`record`] was called (one per system run).
    pub runs: u32,
}

impl Sample {
    /// Average duration per run, in milliseconds.
    pub fn ms_per_run(&self) -> f32 {
        if self.runs == 0 {
            0.0
        } else {
            self.nanos as f32 / 1e6 / self.runs as f32
        }
    }
}

static REGISTRY: Mutex<Option<HashMap<&'static str, Sample>>> = Mutex::new(None);

/// Accumulates `elapsed_ns` under `label` (call once per system run).
pub fn record(label: &'static str, elapsed_ns: u64) {
    let mut guard = REGISTRY.lock().expect("perf registry lock");
    let registry = guard.get_or_insert_with(HashMap::new);
    let entry = registry.entry(label).or_default();
    entry.nanos += elapsed_ns;
    entry.runs += 1;
}

/// Times `body` and records it under `label`; returns the body's result.
pub fn scoped<T>(label: &'static str, body: impl FnOnce() -> T) -> T {
    let start = std::time::Instant::now();
    let result = body();
    record(label, start.elapsed().as_nanos() as u64);
    result
}

/// Drains every accumulated sample; the registry starts empty afterwards.
pub fn snapshot_and_reset() -> Vec<(&'static str, Sample)> {
    let mut guard = REGISTRY.lock().expect("perf registry lock");
    match guard.take() {
        Some(map) => {
            let mut items: Vec<(&'static str, Sample)> = map.into_iter().collect();
            items.sort_by_key(|(label, _)| *label);
            items
        }
        None => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::{record, scoped, snapshot_and_reset};

    #[test]
    fn accumulates_and_drains() {
        // Unique labels: other tests (and demos) may record into the shared
        // registry concurrently, so only these entries are asserted.
        record("probe_a", 1_000);
        record("probe_a", 3_000);
        record("probe_b", 500);

        let samples = snapshot_and_reset();
        let find = |name: &str| {
            samples
                .iter()
                .find(|(label, _)| *label == name)
                .map(|(_, sample)| *sample)
        };
        let a = find("probe_a").expect("probe_a recorded");
        assert_eq!(a.nanos, 4_000);
        assert_eq!(a.runs, 2);
        assert!((a.ms_per_run() - 0.002).abs() < 1e-6);
        assert_eq!(find("probe_b").expect("probe_b recorded").nanos, 500);

        // Draining resets the labels: the next pass must not see them again.
        let after = snapshot_and_reset();
        assert!(after.iter().all(|(label, _)| *label != "probe_a"));
    }

    #[test]
    fn scoped_times_the_body() {
        scoped("probe_c", || {
            record("probe_c", 2_000);
        });
        let samples = snapshot_and_reset();
        let c = samples.iter().find(|(l, _)| *l == "probe_c").unwrap().1;
        assert!(c.nanos >= 2_000, "scoped adds its own measurement on top");
    }
}
