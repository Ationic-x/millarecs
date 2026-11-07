//! Tests for schedule construction, ordering, conflicts, and execution.

use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Condvar, Mutex,
    },
    time::Duration,
};

use millarecs_macros::Component;

use crate::{
    entity::Entity,
    query::filter::{Changed, With},
    resource::ResourceId,
    system::{
        commands::Commands,
        exclusive_system,
        query::Query as SystemQuery,
        resource::{Res, ResMut},
        schedule::{
            executor::ScheduleExecutor, Schedule, ScheduleError, SystemBatchConfigExt, SystemNode,
        },
        System, SystemAccess,
    },
    world::{execution_world::ExecutionWorld, World, WorldError},
};

#[derive(Default, Component, Debug, PartialEq, Eq)]
struct Counter(i32);

#[derive(Default, Component, Debug, PartialEq, Eq)]
struct Step(i32);

#[derive(Default, Component, Debug, PartialEq, Eq)]
struct Marker;

struct MutateNext(bool);

struct ObservedChanges(Vec<usize>);

#[derive(Default)]
struct ProbeState {
    active: usize,
    max_active: usize,
    overlap_observed: bool,
}

#[derive(Default)]
struct ConcurrencyProbe {
    state: Mutex<ProbeState>,
    signal: Condvar,
}

impl ConcurrencyProbe {
    fn observe(&self, timeout: Duration) {
        let mut state = self.state.lock().unwrap();

        state.active += 1;
        state.max_active = state.max_active.max(state.active);

        if state.active >= 2 {
            state.overlap_observed = true;
        }

        self.signal.notify_all();

        let (mut state, _) = self
            .signal
            .wait_timeout_while(state, timeout, |state| !state.overlap_observed)
            .unwrap();

        state.active -= 1;
        self.signal.notify_all();
    }

    fn max_active(&self) -> usize {
        self.state.lock().unwrap().max_active
    }
}

struct ProbeSystem {
    name: &'static str,
    probe: Arc<ConcurrencyProbe>,
    timeout: Duration,
    initialized: bool,
    access: SystemAccess,
}

impl ProbeSystem {
    fn new(
        name: &'static str,
        probe: Arc<ConcurrencyProbe>,
        timeout: Duration,
        access: SystemAccess,
    ) -> Self {
        Self {
            name,
            probe,
            timeout,
            initialized: false,
            access,
        }
    }
}

impl System for ProbeSystem {
    fn name(&self) -> &'static str {
        self.name
    }

    fn initialize(&mut self, _: &mut World) -> Result<(), WorldError> {
        self.initialized = true;
        Ok(())
    }

    fn is_initialized(&self) -> bool {
        self.initialized
    }

    fn access(&self) -> Option<&SystemAccess> {
        self.initialized.then_some(&self.access)
    }

    fn run(&mut self, _: ExecutionWorld<'_>) -> Result<(), WorldError> {
        self.probe.observe(self.timeout);
        Ok(())
    }

    fn has_deferred_work(&self) -> bool {
        false
    }

    fn apply_deferred(&mut self, _: &mut World) -> Result<(), WorldError> {
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum FailureStage {
    Run,
    ApplyDeferred,
}

struct FailingSystem {
    stage: FailureStage,
    discarded: Arc<AtomicBool>,
    initialized: bool,
    access: SystemAccess,
}

impl FailingSystem {
    fn new(stage: FailureStage, discarded: Arc<AtomicBool>) -> Self {
        Self {
            stage,
            discarded,
            initialized: false,
            access: SystemAccess::default(),
        }
    }

    fn error() -> WorldError {
        WorldError::EntityNotFound(Entity::INVALID)
    }
}

impl System for FailingSystem {
    fn name(&self) -> &'static str {
        "failing_system"
    }

    fn initialize(&mut self, _: &mut World) -> Result<(), WorldError> {
        self.initialized = true;
        Ok(())
    }

    fn is_initialized(&self) -> bool {
        self.initialized
    }

    fn access(&self) -> Option<&SystemAccess> {
        self.initialized.then_some(&self.access)
    }

    fn run(&mut self, _: ExecutionWorld<'_>) -> Result<(), WorldError> {
        match self.stage {
            FailureStage::Run => Err(Self::error()),
            FailureStage::ApplyDeferred => Ok(()),
        }
    }

    fn has_deferred_work(&self) -> bool {
        matches!(self.stage, FailureStage::ApplyDeferred)
    }

    fn apply_deferred(&mut self, _: &mut World) -> Result<(), WorldError> {
        match self.stage {
            FailureStage::Run => Ok(()),
            FailureStage::ApplyDeferred => Err(Self::error()),
        }
    }

    fn discard_deferred(&mut self, _: &mut World) {
        self.discarded.store(true, Ordering::SeqCst);
    }
}

fn setup_world() -> World {
    let mut world = World::new();
    world.register_table::<Counter>().unwrap();
    world.register_table::<Step>().unwrap();
    world.register_table::<Marker>().unwrap();
    world
}

fn add_system(schedule: &mut Schedule, system: impl System + 'static) {
    schedule.systems.push(SystemNode {
        system: Box::new(system),
        sets: Vec::new(),
        condition: None,
        label: None,
        before: Vec::new(),
        after: Vec::new(),
    });

    schedule.invalidate_graph();
}

#[test]
fn new_schedule_is_empty_and_uninitialized() {
    let schedule = Schedule::new();

    assert_eq!(schedule.len(), 0);
    assert!(schedule.is_empty());
    assert!(!schedule.is_poisoned());
    assert!(schedule.execution_graph.is_none());
}

#[test]
fn initialize_empty_schedule_marks_it_initialized() {
    let mut world = setup_world();
    let mut schedule = Schedule::new();

    assert!(schedule.execution_graph.is_none());

    schedule.initialize(&mut world).unwrap();

    assert!(schedule.execution_graph.is_some());
}

#[test]
fn initialize_rejects_duplicate_labels() {
    struct Label;
    let mut world = setup_world();

    let mut schedule = Schedule::new();
    schedule.add_system(|| {}).label::<Label>();
    schedule.add_system(|| {}).label::<Label>();

    let error = schedule.initialize(&mut world).unwrap_err();

    assert!(matches!(error, ScheduleError::DuplicateLabel(_)));
}

#[test]
fn initialize_rejects_unknown_dependencies() {
    struct Missing;

    let mut world = setup_world();

    let mut schedule = Schedule::new();
    schedule.add_system(|| {}).after::<Missing>();

    let error = schedule.initialize(&mut world).unwrap_err();

    assert!(matches!(error, ScheduleError::UnknownDependency { .. }));
}

#[test]
fn initialize_rejects_dependency_cycles() {
    struct First;
    struct Second;

    let mut world = setup_world();
    let mut schedule = Schedule::new();
    schedule
        .add_system(|| {})
        .label::<First>()
        .after::<Second>();
    schedule
        .add_system(|| {})
        .label::<Second>()
        .after::<First>();

    let error = schedule.initialize(&mut world).unwrap_err();

    assert!(matches!(error, ScheduleError::DependencyCycle));
}

#[test]
fn initialization_panic_poisons_the_schedule() {
    struct PanickingInitializeSystem;

    impl System for PanickingInitializeSystem {
        fn name(&self) -> &'static str {
            "panicking_initialize_system"
        }

        fn initialize(&mut self, _: &mut World) -> Result<(), WorldError> {
            panic!("intentional initialization panic");
        }

        fn is_initialized(&self) -> bool {
            false
        }

        fn access(&self) -> Option<&SystemAccess> {
            None
        }

        fn run(&mut self, _: ExecutionWorld<'_>) -> Result<(), WorldError> {
            unreachable!("initialization must fail before execution");
        }

        fn has_deferred_work(&self) -> bool {
            false
        }

        fn apply_deferred(&mut self, _: &mut World) -> Result<(), WorldError> {
            Ok(())
        }
    }

    let mut world = setup_world();
    let mut schedule = Schedule::new();
    add_system(&mut schedule, PanickingInitializeSystem);

    let result = catch_unwind(AssertUnwindSafe(|| {
        let _ = schedule.initialize(&mut world);
    }));

    assert!(result.is_err());
    assert!(schedule.is_poisoned());
    assert!(matches!(
        schedule.initialize(&mut world),
        Err(ScheduleError::SchedulePoisoned),
    ));
}

#[test]
fn executor_owns_the_requested_worker_count() {
    let executor = ScheduleExecutor::new(3);

    assert_eq!(executor.worker_count(), 3);
    assert!(!executor.is_poisoned());
}

#[test]
fn independent_systems_run_concurrently() {
    struct FirstResource(usize);
    struct SecondResource(usize);

    let probe = Arc::new(ConcurrencyProbe::default());
    let timeout = Duration::from_secs(1);
    let first_probe = Arc::clone(&probe);
    let second_probe = Arc::clone(&probe);
    let mut world = setup_world();
    let mut schedule = Schedule::new();

    schedule.add_system(move |mut resource: ResMut<FirstResource>| {
        resource.0 += 1;
        first_probe.observe(timeout);
    });

    schedule.add_system(move |mut resource: ResMut<SecondResource>| {
        resource.0 += 1;
        second_probe.observe(timeout);
    });

    world.insert_resource(FirstResource(0));
    world.insert_resource(SecondResource(0));
    let mut executor = ScheduleExecutor::new(2);

    schedule.run(&mut world, &mut executor).unwrap();

    assert_eq!(probe.max_active(), 2);
    assert_eq!(world.get_resource::<FirstResource>().unwrap().0, 1);
    assert_eq!(world.get_resource::<SecondResource>().unwrap().0, 1);
}

#[test]
fn conflicting_systems_do_not_overlap() {
    struct SharedResource;

    let probe = Arc::new(ConcurrencyProbe::default());
    let timeout = Duration::from_millis(50);
    let mut world = setup_world();
    let mut schedule = Schedule::new();

    for name in ["first_conflicting_probe", "second_conflicting_probe"] {
        let access = SystemAccess::from_resource_write(ResourceId::of::<SharedResource>()).unwrap();

        add_system(
            &mut schedule,
            ProbeSystem::new(name, Arc::clone(&probe), timeout, access),
        );
    }

    let mut executor = ScheduleExecutor::new(2);

    schedule.run(&mut world, &mut executor).unwrap();

    assert_eq!(probe.max_active(), 1);
}

#[test]
fn explicit_dependencies_wait_for_predecessors() {
    struct First;

    let order = Arc::new(Mutex::new(Vec::new()));
    let first_order = Arc::clone(&order);
    let second_order = Arc::clone(&order);

    let mut world = setup_world();
    let mut schedule = Schedule::new();

    schedule
        .add_system(move || first_order.lock().unwrap().push("first"))
        .label::<First>();

    schedule
        .add_system(move || {
            let mut order = second_order.lock().unwrap();
            assert_eq!(order.as_slice(), ["first"]);
            order.push("second");
        })
        .after::<First>();

    let mut executor = ScheduleExecutor::new(2);

    schedule.run(&mut world, &mut executor).unwrap();

    assert_eq!(order.lock().unwrap().as_slice(), ["first", "second"]);
}

#[test]
fn execution_state_handles_multiple_bitset_words_and_repeated_runs() {
    const SYSTEM_COUNT: usize = 130;

    let calls = Arc::new(AtomicUsize::new(0));
    let mut world = setup_world();
    let mut schedule = Schedule::new();

    for _ in 0..SYSTEM_COUNT {
        let calls = Arc::clone(&calls);

        schedule.add_system(move || {
            calls.fetch_add(1, Ordering::SeqCst);
        });
    }

    let mut executor = ScheduleExecutor::new(4);

    schedule.run(&mut world, &mut executor).unwrap();
    schedule.run(&mut world, &mut executor).unwrap();

    assert_eq!(calls.load(Ordering::SeqCst), SYSTEM_COUNT * 2);
}

#[test]
fn run_executes_added_system() {
    fn increment(mut query: SystemQuery<&mut Counter>) {
        for mut counter in query.iter_mut() {
            counter.0 += 1;
        }
    }

    let mut world = setup_world();
    let entity = world.spawn((Counter(0),)).unwrap();

    let mut schedule = Schedule::new();
    let mut executor = ScheduleExecutor::default();
    schedule.add_system(increment);

    schedule.run(&mut world, &mut executor).unwrap();

    assert_eq!(world.get::<Counter>(entity), Some(&Counter(1)));
}

#[test]
fn systems_run_in_insertion_order() {
    fn add_step(mut query: SystemQuery<(&mut Counter, &Step)>) {
        for (mut counter, step) in query.iter_mut() {
            counter.0 += step.0;
        }
    }

    fn double(mut query: SystemQuery<&mut Counter>) {
        for mut counter in query.iter_mut() {
            counter.0 *= 2;
        }
    }

    let mut world = setup_world();
    let entity = world.spawn((Counter(2), Step(3))).unwrap();
    let mut schedule = Schedule::new();
    let mut executor = ScheduleExecutor::default();
    schedule.add_systems((add_step, double));

    schedule.run(&mut world, &mut executor).unwrap();

    assert_eq!(world.get::<Counter>(entity), Some(&Counter(10)));
}

#[test]
fn run_can_be_called_repeatedly() {
    fn increment(mut query: SystemQuery<&mut Counter>) {
        for mut counter in query.iter_mut() {
            counter.0 += 1;
        }
    }

    let mut world = setup_world();
    let mut executor = ScheduleExecutor::default();
    let entity = world.spawn((Counter(0),)).unwrap();
    let mut schedule = Schedule::new();
    schedule.add_system(increment);

    for _ in 0..4 {
        schedule.run(&mut world, &mut executor).unwrap();
    }

    assert_eq!(world.get::<Counter>(entity), Some(&Counter(4)));
}

#[test]
fn initialized_query_state_sees_archetypes_created_after_initialization() {
    fn increment(mut query: SystemQuery<&mut Counter>) {
        for mut counter in query.iter_mut() {
            counter.0 += 1;
        }
    }

    let mut world = setup_world();
    let mut executor = ScheduleExecutor::default();
    let mut schedule = Schedule::new();
    schedule.add_system(increment);
    schedule.initialize(&mut world).unwrap();

    let entity = world.spawn((Counter(7),)).unwrap();
    schedule.run(&mut world, &mut executor).unwrap();

    assert_eq!(world.get::<Counter>(entity), Some(&Counter(8)));
}

#[test]
fn schedule_supports_filtered_systems() {
    fn increment_marked(mut query: SystemQuery<&mut Counter, With<Marker>>) {
        for mut counter in query.iter_mut() {
            counter.0 += 10;
        }
    }

    let mut world = setup_world();
    let mut executor = ScheduleExecutor::default();
    let marked = world.spawn((Counter(1), Marker)).unwrap();
    let plain = world.spawn((Counter(1),)).unwrap();
    let mut schedule = Schedule::new();
    schedule.add_system(increment_marked);

    schedule.run(&mut world, &mut executor).unwrap();

    assert_eq!(world.get::<Counter>(marked), Some(&Counter(11)));
    assert_eq!(world.get::<Counter>(plain), Some(&Counter(1)));
}

#[test]
fn changed_filter_tracks_mutable_query_writes() {
    fn maybe_mutate(mut query: SystemQuery<&mut Counter>, mut mutate: ResMut<MutateNext>) {
        if !mutate.0 {
            return;
        }

        for mut counter in query.iter_mut() {
            counter.0 += 1;
        }

        mutate.0 = false;
    }

    fn observe_changed(
        mut query: SystemQuery<Entity, Changed<Counter>>,
        mut observed: ResMut<ObservedChanges>,
    ) {
        observed.0.push(query.iter().count());
    }

    let mut world = setup_world();
    let mut executor = ScheduleExecutor::default();
    world.insert_resource(MutateNext(false));
    world.insert_resource(ObservedChanges(Vec::new()));
    let entity = world.spawn((Counter(0),)).unwrap();

    let mut schedule = Schedule::new();
    schedule.add_systems((maybe_mutate, observe_changed));

    schedule.run(&mut world, &mut executor).unwrap();
    schedule.run(&mut world, &mut executor).unwrap();
    world.get_resource_mut::<MutateNext>().unwrap().0 = true;
    schedule.run(&mut world, &mut executor).unwrap();
    schedule.run(&mut world, &mut executor).unwrap();

    assert_eq!(world.get::<Counter>(entity), Some(&Counter(1)));
    assert_eq!(
        world.get_resource::<ObservedChanges>().unwrap().0,
        vec![1, 0, 1, 0]
    );
}

#[test]
fn initialize_reports_conflicting_system_parameters() {
    fn invalid(_read: SystemQuery<&Counter>, _write: SystemQuery<&mut Counter>) {}

    let mut world = setup_world();
    let mut schedule = Schedule::new();
    schedule.add_system(invalid);

    let err = schedule.initialize(&mut world).unwrap_err();

    match err {
        ScheduleError::SystemInitialize { system, source } => {
            assert!(
                system.contains("invalid"),
                "expected system name to contain `invalid`, got `{system}`"
            );

            assert!(
                matches!(source, WorldError::Query(_)),
                "expected query conflict error, got {source:?}"
            );
        }
        other => panic!("expected SystemInitialize error, got {other:?}"),
    }
}

#[test]
fn add_system_stores_systems_in_order() {
    fn first() {}
    fn second() {}

    let mut schedule = Schedule::new();
    schedule.add_systems((first, second));

    assert_eq!(schedule.systems.len(), 2);
}

#[test]
fn initialize_does_not_require_requested_resource_to_exist() {
    struct Time;

    fn needs_time(_time: Res<Time>) {}

    let mut world = setup_world();
    let mut executor = ScheduleExecutor::default();
    let mut schedule = Schedule::new();
    schedule.add_system(needs_time);

    schedule.initialize(&mut world).unwrap();

    let err = schedule.run(&mut world, &mut executor).unwrap_err();

    assert!(matches!(err, ScheduleError::SystemRun { .. }));
}

#[test]
fn run_materializes_current_resource_values() {
    struct Time {
        delta: i32,
    }

    struct Counter(i32);

    fn update(time: Res<Time>, mut counter: ResMut<Counter>) {
        counter.0 += time.delta;
    }

    let mut world = setup_world();
    let mut executor = ScheduleExecutor::default();
    world.insert_resource(Time { delta: 2 });
    world.insert_resource(Counter(0));

    let mut schedule = Schedule::new();
    schedule.add_system(update);

    schedule.run(&mut world, &mut executor).unwrap();

    world.get_resource_mut::<Time>().unwrap().delta = 5;
    schedule.run(&mut world, &mut executor).unwrap();

    assert_eq!(world.get_resource::<Counter>().unwrap().0, 7);
}

#[test]
fn commands_are_visible_to_explicit_dependents() {
    struct SpawnCounter;

    fn spawn(mut commands: Commands) {
        commands.spawn((Counter(1),));
    }

    fn increment(mut query: SystemQuery<&mut Counter>) {
        for mut counter in query.iter_mut() {
            counter.0 += 1;
        }
    }

    let mut world = setup_world();
    let mut executor = ScheduleExecutor::default();
    let mut schedule = Schedule::new();

    schedule.add_system(spawn).label::<SpawnCounter>();

    schedule.add_system(increment).after::<SpawnCounter>();

    schedule.run(&mut world, &mut executor).unwrap();

    let mut query = world.query_all::<&Counter>().unwrap();

    let values: Vec<_> = query.iter().map(|counter| counter.0).collect();

    assert_eq!(values, vec![2]);
}

#[test]
fn run_failure_discards_deferred_work() {
    let discarded = Arc::new(AtomicBool::new(false));
    let mut world = setup_world();
    let mut schedule = Schedule::new();

    add_system(
        &mut schedule,
        FailingSystem::new(FailureStage::Run, Arc::clone(&discarded)),
    );

    let mut executor = ScheduleExecutor::default();
    let error = schedule.run(&mut world, &mut executor).unwrap_err();

    assert!(matches!(error, ScheduleError::SystemRun { .. }));
    assert!(discarded.load(Ordering::SeqCst));
}

#[test]
fn deferred_application_failure_discards_deferred_work() {
    let discarded = Arc::new(AtomicBool::new(false));
    let mut world = setup_world();
    let mut schedule = Schedule::new();

    add_system(
        &mut schedule,
        FailingSystem::new(FailureStage::ApplyDeferred, Arc::clone(&discarded)),
    );

    let mut executor = ScheduleExecutor::default();
    let error = schedule.run(&mut world, &mut executor).unwrap_err();

    assert!(matches!(error, ScheduleError::SystemApplyDeferred { .. }));
    assert!(discarded.load(Ordering::SeqCst));
}

#[test]
fn run_failure_waits_for_already_dispatched_systems() {
    let discarded = Arc::new(AtomicBool::new(false));
    let completed = Arc::new(AtomicBool::new(false));
    let completed_by_system = Arc::clone(&completed);
    let mut world = setup_world();
    let mut schedule = Schedule::new();

    add_system(
        &mut schedule,
        FailingSystem::new(FailureStage::Run, Arc::clone(&discarded)),
    );

    schedule.add_system(move || {
        std::thread::sleep(Duration::from_millis(25));
        completed_by_system.store(true, Ordering::SeqCst);
    });

    let mut executor = ScheduleExecutor::new(2);
    let error = schedule.run(&mut world, &mut executor).unwrap_err();

    assert!(matches!(error, ScheduleError::SystemRun { .. }));
    assert!(completed.load(Ordering::SeqCst));
    assert!(discarded.load(Ordering::SeqCst));
    assert!(!executor.is_poisoned());
}

#[test]
fn deferred_panic_releases_reservations_and_keeps_executor_usable() {
    fn panic_during_apply(mut commands: Commands) {
        commands.add_infallible(|_| panic!("intentional deferred panic"));
        commands.spawn((Counter(1),));
    }

    fn healthy_system() {}

    let mut world = setup_world();
    let mut executor = ScheduleExecutor::new(2);
    let mut failing_schedule = Schedule::new();
    failing_schedule.add_system(panic_during_apply);

    let result = catch_unwind(AssertUnwindSafe(|| {
        let _ = failing_schedule.run(&mut world, &mut executor);
    }));

    assert!(result.is_err());
    assert_eq!(world.len(), 0);
    assert_eq!(world.entities().reserved_len(), 0);
    assert!(!executor.is_poisoned());

    assert!(matches!(
        failing_schedule.run(&mut world, &mut executor),
        Err(ScheduleError::SchedulePoisoned),
    ));

    let mut healthy_schedule = Schedule::new();
    healthy_schedule.add_system(healthy_system);
    healthy_schedule.run(&mut world, &mut executor).unwrap();
}

#[test]
fn panicking_system_poisons_only_its_schedule() {
    fn panicking_system(mut commands: Commands) {
        commands.spawn((Counter(1),));
        panic!("intentional system panic");
    }

    fn healthy_system() {}

    let mut world = setup_world();
    let mut executor = ScheduleExecutor::new(2);

    let mut failing_schedule = Schedule::new();
    failing_schedule.add_system(panicking_system);

    let result = catch_unwind(AssertUnwindSafe(|| {
        let _ = failing_schedule.run(&mut world, &mut executor);
    }));

    assert!(result.is_err());
    assert_eq!(world.len(), 0);
    assert_eq!(world.entities().reserved_len(), 0);
    assert!(!executor.is_poisoned());

    assert!(matches!(
        failing_schedule.run(&mut world, &mut executor,),
        Err(ScheduleError::SchedulePoisoned),
    ));

    let mut healthy_schedule = Schedule::new();
    healthy_schedule.add_system(healthy_system);

    healthy_schedule.run(&mut world, &mut executor).unwrap();
}

#[test]
fn schedule_rejects_a_different_world_after_binding() {
    let mut first = setup_world();
    let mut second = setup_world();
    let mut schedule = Schedule::new();

    schedule.add_system(|| {});
    schedule.initialize(&mut first).unwrap();

    let result = schedule.initialize(&mut second);

    assert!(matches!(
        result,
        Err(ScheduleError::WorldMismatch {
            expected,
            actual,
        }) if expected == first.id() && actual == second.id()
    ));
}

#[test]
fn batch_after_applies_dependency_to_every_system_in_tuple() {
    struct Producer;

    fn produce(mut commands: Commands) {
        commands.spawn((Counter(1),));
    }

    fn increment(mut query: SystemQuery<&mut Counter>) {
        for mut counter in query.iter_mut() {
            counter.0 += 1;
        }
    }

    fn double(mut query: SystemQuery<&mut Counter>) {
        for mut counter in query.iter_mut() {
            counter.0 *= 2;
        }
    }

    let mut world = setup_world();
    let mut executor = ScheduleExecutor::new(2);
    let mut schedule = Schedule::new();

    schedule.add_system(produce).label::<Producer>();

    schedule.add_systems((increment, double).after::<Producer>());

    schedule.run(&mut world, &mut executor).unwrap();

    let mut query = world.query_all::<&Counter>().unwrap();
    let values: Vec<_> = query.iter().map(|counter| counter.0).collect();

    assert_eq!(values, vec![4]);
}

#[test]
fn batch_after_makes_previous_deferred_work_visible_to_all_systems() {
    struct Producer;

    #[derive(Default)]
    struct SawCounterA(bool);

    #[derive(Default)]
    struct SawCounterB(bool);

    fn produce(mut commands: Commands) {
        commands.spawn((Counter(1),));
    }

    fn observe_a(mut query: SystemQuery<&Counter>, mut seen: ResMut<SawCounterA>) {
        seen.0 = query.iter().next().is_some();
    }

    fn observe_b(mut query: SystemQuery<&Counter>, mut seen: ResMut<SawCounterB>) {
        seen.0 = query.iter().next().is_some();
    }

    let mut world = setup_world();
    world.insert_resource(SawCounterA::default());
    world.insert_resource(SawCounterB::default());

    let mut executor = ScheduleExecutor::new(2);
    let mut schedule = Schedule::new();

    schedule.add_system(produce).label::<Producer>();

    schedule.add_systems((observe_a, observe_b).after::<Producer>());

    schedule.run(&mut world, &mut executor).unwrap();

    assert!(world.get_resource::<SawCounterA>().unwrap().0);
    assert!(world.get_resource::<SawCounterB>().unwrap().0);
}

#[test]
fn deferred_barrier_makes_previous_commands_visible_to_later_systems() {
    fn spawn(mut commands: Commands) {
        commands.spawn((Counter(1),));
    }

    fn observe(mut query: SystemQuery<&Counter>, mut observed: ResMut<Vec<i32>>) {
        observed.extend(query.iter().map(|counter| counter.0));
    }

    let mut world = setup_world();
    world.insert_resource(Vec::<i32>::new());

    let mut schedule = Schedule::new();

    schedule.add_system(spawn);
    schedule.add_deferred_barrier();
    schedule.add_system(observe);

    let mut executor = ScheduleExecutor::new(2);
    schedule.run(&mut world, &mut executor).unwrap();

    assert_eq!(world.get_resource::<Vec<i32>>().unwrap().as_slice(), &[1]);
}

#[test]
fn multiple_deferred_barriers_split_schedule_into_ordered_segments() {
    fn spawn_one(mut commands: Commands) {
        commands.spawn((Counter(1),));
    }

    fn spawn_two(mut commands: Commands) {
        commands.spawn((Counter(2),));
    }

    fn observe(mut query: SystemQuery<&Counter>, mut observed: ResMut<Vec<i32>>) {
        let mut values = query.iter().map(|counter| counter.0).collect::<Vec<_>>();
        values.sort();

        observed.clear();
        observed.extend(values);
    }

    let mut world = setup_world();
    world.insert_resource(Vec::<i32>::new());

    let mut schedule = Schedule::new();

    schedule.add_system(spawn_one);
    schedule.add_deferred_barrier();

    schedule.add_system(spawn_two);
    schedule.add_deferred_barrier();

    schedule.add_system(observe);

    let mut executor = ScheduleExecutor::new(2);
    schedule.run(&mut world, &mut executor).unwrap();

    assert_eq!(
        world.get_resource::<Vec<i32>>().unwrap().as_slice(),
        &[1, 2]
    );
}

#[test]
fn deferred_barrier_works_between_system_batches() {
    fn spawn_a(mut commands: Commands) {
        commands.spawn((Counter(10),));
    }

    fn spawn_b(mut commands: Commands) {
        commands.spawn((Counter(20),));
    }

    fn observe_a(mut query: SystemQuery<&Counter>, mut observed: ResMut<Vec<i32>>) {
        observed.extend(query.iter().map(|counter| counter.0));
    }

    fn observe_b(mut query: SystemQuery<&Counter>, mut observed: ResMut<Vec<i32>>) {
        observed.extend(query.iter().map(|counter| counter.0));
    }

    let mut world = setup_world();
    world.insert_resource(Vec::<i32>::new());

    let mut schedule = Schedule::new();

    schedule.add_systems((spawn_a, spawn_b));
    schedule.add_deferred_barrier();
    schedule.add_systems((observe_a, observe_b));

    let mut executor = ScheduleExecutor::new(4);
    schedule.run(&mut world, &mut executor).unwrap();

    let mut values = world.get_resource::<Vec<i32>>().unwrap().clone();
    values.sort();

    assert_eq!(values, vec![10, 10, 20, 20]);
}

#[test]
fn chained_systems_run_in_tuple_order_without_access_conflicts() {
    use std::{
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        },
        time::Duration,
    };

    let step = Arc::new(AtomicUsize::new(0));

    let first_step = Arc::clone(&step);
    let second_step = Arc::clone(&step);
    let third_step = Arc::clone(&step);

    let first = move || {
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(first_step.fetch_add(1, Ordering::SeqCst), 0);
    };

    let second = move || {
        assert_eq!(second_step.fetch_add(1, Ordering::SeqCst), 1);
    };

    let third = move || {
        assert_eq!(third_step.fetch_add(1, Ordering::SeqCst), 2);
    };

    let mut world = setup_world();
    let mut schedule = Schedule::new();

    schedule.add_systems((first, second, third).chain());

    let mut executor = ScheduleExecutor::new(3);
    schedule.run(&mut world, &mut executor).unwrap();

    assert_eq!(step.load(Ordering::SeqCst), 3);
}

#[test]
fn chained_system_observes_previous_deferred_work() {
    fn spawn(mut commands: Commands) {
        commands.spawn((Counter(1),));
    }

    fn increment(mut query: SystemQuery<&mut Counter>) {
        for mut counter in query.iter_mut() {
            counter.0 += 1;
        }
    }

    let mut world = setup_world();
    let mut schedule = Schedule::new();

    schedule.add_systems((spawn, increment).chain());

    let mut executor = ScheduleExecutor::new(2);
    schedule.run(&mut world, &mut executor).unwrap();

    let mut query = world.query_all::<&Counter>().unwrap();
    let values: Vec<_> = query.iter().map(|counter| counter.0).collect();

    assert_eq!(values, vec![2]);
}

#[test]
fn chained_batch_runs_after_external_label() {
    struct Producer;

    fn produce(mut commands: Commands) {
        commands.spawn((Counter(1),));
    }

    fn increment(mut query: SystemQuery<&mut Counter>) {
        for mut counter in query.iter_mut() {
            counter.0 += 1;
        }
    }

    fn double(mut query: SystemQuery<&mut Counter>) {
        for mut counter in query.iter_mut() {
            counter.0 *= 2;
        }
    }

    let mut world = setup_world();
    let mut schedule = Schedule::new();

    schedule.add_system(produce).label::<Producer>();
    schedule.add_systems((increment, double).chain().after::<Producer>());

    let mut executor = ScheduleExecutor::new(3);
    schedule.run(&mut world, &mut executor).unwrap();

    let mut query = world.query_all::<&Counter>().unwrap();
    let values: Vec<_> = query.iter().map(|counter| counter.0).collect();

    assert_eq!(values, vec![4]);
}

#[test]
fn chained_batch_runs_before_external_label() {
    struct Observer;

    fn increment(mut query: SystemQuery<&mut Counter>) {
        for mut counter in query.iter_mut() {
            counter.0 += 1;
        }
    }

    fn double(mut query: SystemQuery<&mut Counter>) {
        for mut counter in query.iter_mut() {
            counter.0 *= 2;
        }
    }

    fn observe(mut query: SystemQuery<&Counter>, mut observed: ResMut<Vec<i32>>) {
        observed.extend(query.iter().map(|counter| counter.0));
    }

    let mut world = setup_world();
    world.spawn((Counter(1),)).unwrap();
    world.insert_resource(Vec::<i32>::new());

    let mut schedule = Schedule::new();

    schedule.add_system(observe).label::<Observer>();
    schedule.add_systems((increment, double).chain().before::<Observer>());

    let mut executor = ScheduleExecutor::new(3);
    schedule.run(&mut world, &mut executor).unwrap();

    assert_eq!(world.get_resource::<Vec<i32>>().unwrap().as_slice(), &[4]);
}

#[test]
fn command_spawns_publish_spawned_events_to_observers() {
    #[derive(Default)]
    struct SpawnedObserver;
    struct Observed(Vec<Entity>);

    fn spawner(mut commands: Commands) {
        commands.spawn((Counter(0),));
    }

    fn observe_spawned(
        mut reader: crate::system::event::EventReader<crate::event::Spawned>,
        mut observed: ResMut<Observed>,
    ) {
        for event in reader.read() {
            observed.0.push(event.entity);
        }
    }

    let mut world = World::new();
    world.register_table::<Counter>().unwrap();
    world.observe_spawned();
    world.insert_resource(Observed(Vec::new()));

    let mut schedule = Schedule::new();
    schedule.add_system(spawner).label::<SpawnedObserver>();
    schedule.add_deferred_barrier();
    schedule
        .add_system(observe_spawned)
        .after::<SpawnedObserver>();

    let mut executor = ScheduleExecutor::new(2);
    schedule.run(&mut world, &mut executor).unwrap();

    assert_eq!(world.get_resource::<Observed>().unwrap().0.len(), 1);
}

#[test]
fn mutated_batches_are_published_once_per_cycle_after_writes() {
    #[derive(Default)]
    struct MutatedObserver;

    fn write(mut query: SystemQuery<&mut Counter>) {
        for mut counter in query.iter_mut() {
            counter.0 += 1;
        }
    }

    fn observe_mutated(
        mut reader: crate::system::event::EventReader<crate::event::Mutated<Counter>>,
        mut observed: ResMut<ObservedChanges>,
    ) {
        for event in reader.read() {
            observed.0.push(event.entities.len());
        }
    }

    let mut world = World::new();
    world.register_table::<Counter>().unwrap();
    for row in 0..3 {
        world.spawn((Counter(row),)).unwrap();
    }
    world.observe_changed::<Counter>().unwrap();
    world.insert_resource(ObservedChanges(Vec::new()));

    let mut schedule = Schedule::new();
    schedule.add_system(write).label::<MutatedObserver>();
    schedule.add_deferred_barrier();
    schedule
        .add_system(observe_mutated)
        .after::<MutatedObserver>();

    let mut executor = ScheduleExecutor::new(2);
    schedule.run(&mut world, &mut executor).unwrap();
    schedule.run(&mut world, &mut executor).unwrap();

    // Cycle 2's start-of-cycle dispatch publishes the writes from cycle 1.
    assert_eq!(
        world
            .get_resource::<ObservedChanges>()
            .unwrap()
            .0
            .as_slice(),
        &[3]
    );
}

#[test]
fn exclusive_systems_run_inline_between_phases_on_both_executors() {
    struct Victim(Entity);
    struct Survivors(u32);

    fn write(mut query: SystemQuery<&mut Counter>) {
        for mut counter in query.iter_mut() {
            counter.0 += 1;
        }
    }

    fn despawn_victim(world: &mut World) -> Result<(), WorldError> {
        let victim = world.get_resource::<Victim>().unwrap().0;
        world.despawn(victim)?;
        world.insert_resource(Survivors(world.len() as u32));
        Ok(())
    }

    fn run_scenario(executor: &mut ScheduleExecutor) -> (u32, u32, Vec<i32>, bool) {
        let mut world = World::new();
        world.register_table::<Counter>().unwrap();

        let victim = world.spawn((Counter(0),)).unwrap();
        world.spawn((Counter(0),)).unwrap();
        world.spawn((Counter(0),)).unwrap();
        world.insert_resource(Victim(victim));

        let mut schedule = Schedule::new();
        schedule.add_system(write);
        schedule.add_system(exclusive_system(despawn_victim));
        schedule.add_system(write);

        schedule.run(&mut world, executor).unwrap();

        let survivors = world.get_resource::<Survivors>().unwrap().0;
        let total = world.len() as u32;
        let mut counters: Vec<i32> = world
            .query::<&Counter, ()>()
            .unwrap()
            .iter()
            .map(|counter| counter.0)
            .collect();
        counters.sort_unstable();

        (survivors, total, counters, world.contains(victim))
    }

    let mut single = ScheduleExecutor::single_threaded();
    let (survivors, total, counters, victim_alive) = run_scenario(&mut single);

    assert_eq!(survivors, 2, "exclusive despawn leaves two entities");
    assert_eq!(total, 2);
    assert_eq!(counters, vec![2, 2], "both writes reach the survivors");
    assert!(!victim_alive);

    let mut multi = ScheduleExecutor::new(4);
    let (survivors, total, counters, victim_alive) = run_scenario(&mut multi);

    assert_eq!(survivors, 2, "exclusive despawn leaves two entities");
    assert_eq!(total, 2);
    assert_eq!(counters, vec![2, 2], "both writes reach the survivors");
    assert!(!victim_alive);
}
