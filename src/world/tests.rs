//! Tests for the `World` API: entity lifecycle, table and sparse component
//! storage, fragmented-archetype compaction, queries and filters, resources,
//! and scheduled systems.

use millarecs_macros::Component;

use super::*;
use crate::{
    __private::BundleError,
    component::archetype::ARCHETYPE_PAGE_ROWS,
    query::{
        filter::{Added, AnyOf, Changed, Not, Or, With, Without},
        QueryError,
    },
    system::{
        param::SystemParam,
        resource::{Res, ResMut},
        schedule::{Schedule, ScheduleError, ScheduleExecutor},
    },
};

#[derive(Default, Component, Debug, PartialEq, Eq)]
struct Position {
    x: i32,
    y: i32,
}

#[derive(Default, Component, Debug, PartialEq, Eq)]
struct Velocity {
    x: i32,
    y: i32,
}

#[derive(Default, Component, Debug, PartialEq, Eq)]
struct Health(i32);

#[derive(Default, Component, Debug, PartialEq, Eq)]
struct Mana(i32);

#[derive(Default, Component, Debug, PartialEq, Eq)]
struct Name(String);

#[derive(Default, Component, Debug, PartialEq, Eq)]
struct Player;

#[derive(Default, Component, Debug, PartialEq, Eq)]
struct Enemy;

#[derive(Default, Component, Debug, PartialEq, Eq)]
struct Frozen;

#[derive(Default, Component, Debug, PartialEq, Eq)]
struct Unregistered;

#[derive(Component, Debug, PartialEq, Eq)]
struct NonDefault(String);

struct DropCounter {
    drops: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl Component for DropCounter {}

impl Drop for DropCounter {
    fn drop(&mut self) {
        self.drops.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

#[derive(Component, Debug, PartialEq, Eq)]
enum MotionState {
    Moving,
}

fn register_components(world: &mut World) {
    world.register_table::<Position>().unwrap();
    world.register_table::<Velocity>().unwrap();
    world.register_table::<Mana>().unwrap();
    world.register_table::<Name>().unwrap();
    world.register_table::<Player>().unwrap();
    world.register_table::<Enemy>().unwrap();
    world.register_sparse::<Health>().unwrap();
    world.register_sparse::<Frozen>().unwrap();
}

fn sorted(mut values: Vec<i32>) -> Vec<i32> {
    values.sort_unstable();
    values
}

#[test]
fn new_world_starts_without_live_entities() {
    let world = World::new();

    assert_eq!(world.len(), 0);
    assert!(world.is_empty());
}

#[test]
fn try_new_creates_the_empty_archetype() {
    let world = World::try_new().unwrap();

    assert_eq!(world.components().archetype_count(), 1);
}

#[test]
fn components_do_not_require_clone_or_default() {
    let mut world = World::new();
    world.register_table::<NonDefault>().unwrap();
    world.register_sparse::<MotionState>().unwrap();

    let entity = world
        .spawn((NonDefault("owned".to_owned()), MotionState::Moving))
        .unwrap();

    assert_eq!(
        world.get::<NonDefault>(entity),
        Some(&NonDefault("owned".to_owned()))
    );
    assert_eq!(world.get::<MotionState>(entity), Some(&MotionState::Moving));
}

#[test]
fn table_components_drop_from_contiguous_and_fragmented_storage() {
    let drops = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

    {
        let mut world = World::new();
        world.register_table::<DropCounter>().unwrap();

        world
            .spawn((DropCounter {
                drops: drops.clone(),
            },))
            .unwrap();
        world
            .spawn_with_storage(
                ArchetypeStorageKind::Fragmented,
                (DropCounter {
                    drops: drops.clone(),
                },),
            )
            .unwrap();

        let removed = world
            .spawn_with_storage(
                ArchetypeStorageKind::Fragmented,
                (DropCounter {
                    drops: drops.clone(),
                },),
            )
            .unwrap();

        world.despawn(removed).unwrap();
        assert_eq!(drops.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    assert_eq!(drops.load(std::sync::atomic::Ordering::SeqCst), 3);
}

#[test]
fn component_registration_returns_stable_component_ids() {
    let mut world = World::new();

    let position = world.register_table::<Position>().unwrap();
    let health = world.register_sparse::<Health>().unwrap();

    assert_ne!(position, health);
    assert!(world.components().id_of::<Position>().is_some());
    assert!(world.components().id_of::<Health>().is_some());
}

#[test]
fn empty_spawns_are_distinct_and_alive() {
    let mut world = World::new();
    world.register_table::<Position>().unwrap();

    let first = world.spawn_empty().unwrap();
    let second = world.spawn_empty().unwrap();

    assert_ne!(first, second);
    assert!(world.contains(first));
    assert!(world.contains(second));
    assert_eq!(world.len(), 2);
    assert!(!world.is_empty());
    assert_eq!(world.get::<Position>(first), None);
    assert_eq!(
        world.entities.location(first).unwrap().storage,
        ArchetypeStorageKind::Contiguous
    );
}

#[test]
fn spawn_applies_table_and_sparse_components() {
    let mut world = World::new();
    register_components(&mut world);

    let entity = world
        .spawn((Position { x: 1, y: 2 }, Velocity { x: 3, y: 4 }, Health(90)))
        .unwrap();

    assert_eq!(
        world.get::<Position>(entity),
        Some(&Position { x: 1, y: 2 })
    );
    assert_eq!(
        world.get::<Velocity>(entity),
        Some(&Velocity { x: 3, y: 4 })
    );
    assert_eq!(world.get::<Health>(entity), Some(&Health(90)));
    assert_eq!(
        world.entities.location(entity).unwrap().storage,
        ArchetypeStorageKind::Contiguous
    );
}

#[test]
fn spawn_reuses_plan_and_preserves_original_bundle_order() {
    let mut world = World::new();
    register_components(&mut world);

    let first = world
        .spawn((Health(10), Velocity { x: 3, y: 4 }, Position { x: 1, y: 2 }))
        .unwrap();
    let plans_after_first_spawn = world.components.spawn_plan_count();
    let second = world
        .spawn_with_storage(
            ArchetypeStorageKind::Fragmented,
            (Health(20), Velocity { x: 7, y: 8 }, Position { x: 5, y: 6 }),
        )
        .unwrap();

    assert_eq!(plans_after_first_spawn, 1);
    assert_eq!(world.components.spawn_plan_count(), 1);
    assert_eq!(world.get::<Health>(first), Some(&Health(10)));
    assert_eq!(world.get::<Position>(first), Some(&Position { x: 1, y: 2 }));
    assert_eq!(world.get::<Velocity>(first), Some(&Velocity { x: 3, y: 4 }));
    assert_eq!(world.get::<Health>(second), Some(&Health(20)));
    assert_eq!(
        world.get::<Position>(second),
        Some(&Position { x: 5, y: 6 })
    );
    assert_eq!(
        world.get::<Velocity>(second),
        Some(&Velocity { x: 7, y: 8 })
    );
    assert_eq!(
        world.entities.location(second).unwrap().storage,
        ArchetypeStorageKind::Fragmented
    );
}

#[test]
fn no_stray_rows_in_the_empty_archetype() {
    let mut world = World::new();
    world.register_table::<Position>().unwrap();

    let entity = world.spawn((Position { x: 8, y: 9 },)).unwrap();

    let empty = world
        .components()
        .get_archetype(world.empty_archetype)
        .unwrap();

    assert_eq!(empty.len(), 0);
    assert_eq!(
        world.get::<Position>(entity),
        Some(&Position { x: 8, y: 9 })
    );
}

#[test]
fn spawn_supports_sparse_only_bundles() {
    let mut world = World::new();
    register_components(&mut world);

    let entity = world.spawn((Health(25),)).unwrap();

    assert_eq!(world.get::<Health>(entity), Some(&Health(25)));
    assert_eq!(world.get::<Position>(entity), None);
}

#[test]
fn failed_spawn_rolls_back_the_reserved_entity() {
    let mut world = World::new();
    world.register_table::<Position>().unwrap();

    let result = world.spawn((Position { x: 1, y: 2 }, Unregistered));

    assert!(matches!(
        result,
        Err(WorldError::Bundle(BundleError::ComponentNotRegistered(_)))
    ));
    assert_eq!(world.len(), 0);
    assert!(world.is_empty());
}

#[test]
fn insert_table_component_migrates_entity_and_keeps_previous_components() {
    let mut world = World::new();
    register_components(&mut world);

    let entity = world.spawn((Position { x: 4, y: 5 }, Health(70))).unwrap();
    world.insert(entity, (Velocity { x: 1, y: 2 },)).unwrap();

    assert_eq!(
        world.get::<Position>(entity),
        Some(&Position { x: 4, y: 5 })
    );
    assert_eq!(
        world.get::<Velocity>(entity),
        Some(&Velocity { x: 1, y: 2 })
    );
    assert_eq!(world.get::<Health>(entity), Some(&Health(70)));
}

#[test]
fn insert_reuses_transition_plan_and_preserves_original_bundle_order() {
    let mut world = World::new();
    register_components(&mut world);

    let first = world.spawn((Position { x: 1, y: 2 },)).unwrap();
    let second = world.spawn((Position { x: 3, y: 4 },)).unwrap();

    world
        .insert(
            first,
            (Health(10), Velocity { x: 5, y: 6 }, Position { x: 7, y: 8 }),
        )
        .unwrap();
    let plans_after_first_insert = world.components.transition_plan_count();

    world
        .insert(
            second,
            (
                Health(20),
                Velocity { x: 9, y: 10 },
                Position { x: 11, y: 12 },
            ),
        )
        .unwrap();

    assert_eq!(plans_after_first_insert, 1);
    assert_eq!(world.components.transition_plan_count(), 1);
    assert_eq!(world.get::<Health>(first), Some(&Health(10)));
    assert_eq!(world.get::<Velocity>(first), Some(&Velocity { x: 5, y: 6 }));
    assert_eq!(world.get::<Position>(first), Some(&Position { x: 7, y: 8 }));
    assert_eq!(world.get::<Health>(second), Some(&Health(20)));
    assert_eq!(
        world.get::<Velocity>(second),
        Some(&Velocity { x: 9, y: 10 })
    );
    assert_eq!(
        world.get::<Position>(second),
        Some(&Position { x: 11, y: 12 })
    );
}

#[test]
fn in_place_transition_drops_replaced_table_value_once() {
    let drops = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

    {
        let mut world = World::new();
        world.register_table::<DropCounter>().unwrap();
        let entity = world
            .spawn((DropCounter {
                drops: drops.clone(),
            },))
            .unwrap();

        world
            .insert(
                entity,
                (DropCounter {
                    drops: drops.clone(),
                },),
            )
            .unwrap();

        assert_eq!(drops.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    assert_eq!(drops.load(std::sync::atomic::Ordering::SeqCst), 2);
}

#[test]
fn transition_plan_migrates_fragmented_rows() {
    let mut world = World::new();
    register_components(&mut world);

    let entity = world
        .spawn_with_storage(
            ArchetypeStorageKind::Fragmented,
            (Position { x: 1, y: 2 }, Health(30)),
        )
        .unwrap();

    world.insert(entity, (Velocity { x: 3, y: 4 },)).unwrap();

    assert_eq!(
        world.entities.location(entity).unwrap().storage,
        ArchetypeStorageKind::Fragmented
    );
    assert_eq!(
        world.get::<Position>(entity),
        Some(&Position { x: 1, y: 2 })
    );
    assert_eq!(
        world.get::<Velocity>(entity),
        Some(&Velocity { x: 3, y: 4 })
    );
    assert_eq!(world.get::<Health>(entity), Some(&Health(30)));
}

#[test]
fn remove_component_reuses_structural_plan_for_both_storage_lanes() {
    let drops = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let mut world = World::new();
    world.register_table::<Position>().unwrap();
    world.register_table::<DropCounter>().unwrap();

    let contiguous = world
        .spawn((
            Position { x: 1, y: 2 },
            DropCounter {
                drops: drops.clone(),
            },
        ))
        .unwrap();
    let fragmented = world
        .spawn_with_storage(
            ArchetypeStorageKind::Fragmented,
            (
                Position { x: 3, y: 4 },
                DropCounter {
                    drops: drops.clone(),
                },
            ),
        )
        .unwrap();

    world.remove::<DropCounter>(contiguous).unwrap();
    let plans_after_first_remove = world.components.transition_plan_count();
    world.remove::<DropCounter>(fragmented).unwrap();

    assert_eq!(plans_after_first_remove, 1);
    assert_eq!(world.components.transition_plan_count(), 1);
    assert_eq!(drops.load(std::sync::atomic::Ordering::SeqCst), 2);
    assert!(world.get::<DropCounter>(contiguous).is_none());
    assert!(world.get::<DropCounter>(fragmented).is_none());
    assert_eq!(
        world.entities.location(fragmented).unwrap().storage,
        ArchetypeStorageKind::Fragmented
    );
    assert_eq!(
        world.get::<Position>(fragmented),
        Some(&Position { x: 3, y: 4 })
    );
}

#[test]
fn remove_bundle_reuses_structural_plan_and_cached_sparse_layout() {
    let mut world = World::new();
    register_components(&mut world);

    let first = world
        .spawn((
            Position { x: 1, y: 2 },
            Velocity { x: 3, y: 4 },
            Health(10),
            Frozen,
        ))
        .unwrap();
    let second = world
        .spawn((
            Position { x: 5, y: 6 },
            Velocity { x: 7, y: 8 },
            Health(20),
            Frozen,
        ))
        .unwrap();

    world.remove_bundle::<(Velocity, Health)>(first).unwrap();
    let plans_after_first_remove = world.components.transition_plan_count();
    world.remove_bundle::<(Velocity, Health)>(second).unwrap();

    assert_eq!(plans_after_first_remove, 1);
    assert_eq!(world.components.transition_plan_count(), 1);
    for entity in [first, second] {
        assert!(world.get::<Position>(entity).is_some());
        assert_eq!(world.get::<Velocity>(entity), None);
        assert_eq!(world.get::<Health>(entity), None);
        assert_eq!(world.get::<Frozen>(entity), Some(&Frozen));
    }
}

#[test]
fn retain_bundle_reuses_structural_plan_and_preserves_selected_sparse_values() {
    let mut world = World::new();
    register_components(&mut world);

    let first = world
        .spawn((
            Position { x: 1, y: 2 },
            Velocity { x: 3, y: 4 },
            Health(10),
            Frozen,
        ))
        .unwrap();
    let second = world
        .spawn((
            Position { x: 5, y: 6 },
            Velocity { x: 7, y: 8 },
            Health(20),
            Frozen,
        ))
        .unwrap();

    world.retain_bundle::<(Position, Health)>(first).unwrap();
    let plans_after_first_retain = world.components.transition_plan_count();
    world.retain_bundle::<(Position, Health)>(second).unwrap();

    assert_eq!(plans_after_first_retain, 1);
    assert_eq!(world.components.transition_plan_count(), 1);
    assert_eq!(world.get::<Position>(first), Some(&Position { x: 1, y: 2 }));
    assert_eq!(
        world.get::<Position>(second),
        Some(&Position { x: 5, y: 6 })
    );
    assert_eq!(world.get::<Health>(first), Some(&Health(10)));
    assert_eq!(world.get::<Health>(second), Some(&Health(20)));

    for entity in [first, second] {
        assert_eq!(world.get::<Velocity>(entity), None);
        assert_eq!(world.get::<Frozen>(entity), None);
    }
}

#[test]
fn insert_sparse_component_keeps_archetype_components_available() {
    let mut world = World::new();
    register_components(&mut world);

    let entity = world
        .spawn((Position { x: 8, y: 9 }, Velocity { x: 1, y: 1 }))
        .unwrap();
    world.insert(entity, (Health(40),)).unwrap();

    assert_eq!(
        world.get::<Position>(entity),
        Some(&Position { x: 8, y: 9 })
    );
    assert_eq!(
        world.get::<Velocity>(entity),
        Some(&Velocity { x: 1, y: 1 })
    );
    assert_eq!(world.get::<Health>(entity), Some(&Health(40)));
}

#[test]
fn insert_sparse_component_replaces_existing_value() {
    let mut world = World::new();
    register_components(&mut world);

    let entity = world.spawn((Health(10),)).unwrap();
    world.insert(entity, (Health(100),)).unwrap();

    assert_eq!(world.get::<Health>(entity), Some(&Health(100)));
}

#[test]
fn insert_unregistered_component_returns_error_without_despawning_entity() {
    let mut world = World::new();
    world.register_table::<Position>().unwrap();

    let entity = world.spawn((Position { x: 3, y: 7 },)).unwrap();
    let result = world.insert(entity, (Unregistered,));

    assert!(matches!(
        result,
        Err(WorldError::Bundle(BundleError::ComponentNotRegistered(_)))
    ));
    assert!(world.contains(entity));
    assert_eq!(
        world.get::<Position>(entity),
        Some(&Position { x: 3, y: 7 })
    );
}

#[test]
fn insert_invalid_entity_reports_entity_not_found() {
    let mut world = World::new();
    world.register_table::<Position>().unwrap();

    let result = world.insert(Entity::INVALID, (Position { x: 1, y: 1 },));

    assert!(matches!(
        result,
        Err(WorldError::EntityNotFound(entity)) if entity == Entity::INVALID
    ));
}

#[test]
fn get_mut_updates_table_and_sparse_components() {
    let mut world = World::new();
    register_components(&mut world);

    let entity = world.spawn((Position { x: 1, y: 2 }, Health(5))).unwrap();

    world.get_mut::<Position>(entity).unwrap().x += 10;
    world.get_mut::<Health>(entity).unwrap().0 += 20;

    assert_eq!(
        world.get::<Position>(entity),
        Some(&Position { x: 11, y: 2 })
    );
    assert_eq!(world.get::<Health>(entity), Some(&Health(25)));
}

#[test]
fn despawn_removes_entity_and_all_components() {
    let mut world = World::new();
    register_components(&mut world);

    let entity = world
        .spawn((Position { x: 1, y: 2 }, Velocity { x: 3, y: 4 }, Health(50)))
        .unwrap();

    world.despawn(entity).unwrap();

    assert!(!world.contains(entity));
    assert_eq!(world.len(), 0);
    assert_eq!(world.get::<Position>(entity), None);
    assert_eq!(world.get::<Velocity>(entity), None);
    assert_eq!(world.get::<Health>(entity), None);
}

#[test]
fn despawn_invalid_or_dead_entity_returns_not_found() {
    let mut world = World::new();

    assert!(matches!(
        world.despawn(Entity::INVALID),
        Err(WorldError::EntityNotFound(_))
    ));

    let entity = world.spawn_empty().unwrap();
    world.despawn(entity).unwrap();

    assert!(matches!(
        world.despawn(entity),
        Err(WorldError::EntityNotFound(_))
    ));
}

#[test]
fn despawn_keeps_swap_removed_entity_addressable() {
    let mut world = World::new();
    register_components(&mut world);

    let first = world
        .spawn((Position { x: 1, y: 0 }, Velocity { x: 10, y: 0 }))
        .unwrap();
    let second = world
        .spawn((Position { x: 2, y: 0 }, Velocity { x: 20, y: 0 }))
        .unwrap();

    world.despawn(first).unwrap();

    assert!(!world.contains(first));
    assert!(world.contains(second));
    assert_eq!(
        world.get::<Position>(second),
        Some(&Position { x: 2, y: 0 })
    );
    assert_eq!(
        world.get::<Velocity>(second),
        Some(&Velocity { x: 20, y: 0 })
    );

    world.get_mut::<Position>(second).unwrap().x = 30;
    assert_eq!(
        world.get::<Position>(second),
        Some(&Position { x: 30, y: 0 })
    );
}

#[test]
fn query_returns_registered_table_components() {
    let mut world = World::new();
    register_components(&mut world);

    world.spawn((Position { x: 2, y: 0 },)).unwrap();
    world.spawn((Position { x: 1, y: 0 }, Health(1))).unwrap();

    let mut query = world.query::<&Position, ()>().unwrap();
    let values: Vec<_> = query.iter().map(|position| position.x).collect();

    assert_eq!(sorted(values), vec![1, 2]);
}

#[test]
fn paged_archetype_reuses_freed_rows_without_moving_other_entities() {
    let mut world = World::new();
    world.register_table::<Position>().unwrap();

    let entities = (0..(ARCHETYPE_PAGE_ROWS + 6))
        .map(|index| {
            world
                .spawn_with_storage(
                    ArchetypeStorageKind::Fragmented,
                    (Position {
                        x: index as i32,
                        y: 0,
                    },),
                )
                .unwrap()
        })
        .collect::<Vec<_>>();

    let removed = entities[3];
    let stable = entities[ARCHETYPE_PAGE_ROWS + 2];
    let removed_location = world.entities.location(removed).unwrap();
    let stable_location = world.entities.location(stable).unwrap();

    world.despawn(removed).unwrap();

    assert_eq!(world.entities.location(stable), Some(stable_location));

    let replacement = world
        .spawn_with_storage(
            ArchetypeStorageKind::Fragmented,
            (Position { x: 999, y: 0 },),
        )
        .expect("replacement spawn should reuse a free archetype row");

    assert_eq!(
        world.entities.location(replacement).unwrap().row,
        removed_location.row
    );

    let mut query = world.query_all::<&Position>().unwrap();
    let values = query.iter().map(|position| position.x).collect::<Vec<_>>();

    assert!(!values.contains(&3));
    assert!(values.contains(&999));
    assert_eq!(values.len(), ARCHETYPE_PAGE_ROWS + 6);
}

#[test]
fn compact_archetypes_moves_live_rows_and_updates_locations() {
    let mut world = World::new();
    world.register_table::<Position>().unwrap();

    let entities = (0..(ARCHETYPE_PAGE_ROWS * 2))
        .map(|index| {
            world
                .spawn_with_storage(
                    ArchetypeStorageKind::Fragmented,
                    (Position {
                        x: index as i32,
                        y: 0,
                    },),
                )
                .unwrap()
        })
        .collect::<Vec<_>>();

    for entity in entities.iter().take(ARCHETYPE_PAGE_ROWS / 2) {
        world.despawn(*entity).unwrap();
    }

    world.compact_archetypes();

    let live = entities
        .iter()
        .skip(ARCHETYPE_PAGE_ROWS / 2)
        .copied()
        .collect::<Vec<_>>();

    for (expected_row, entity) in live.iter().enumerate() {
        let location = world.entities.location(*entity).unwrap();
        assert_eq!(location.row as usize, expected_row);
        assert!(world.get::<Position>(*entity).is_some());
    }

    let mut query = world.query_all::<&Position>().unwrap();
    assert_eq!(query.iter().count(), live.len());
}

#[test]
fn fragmented_compaction_policy_controls_when_pages_are_rebuilt() {
    let mut world = World::new();
    world.register_table::<Position>().unwrap();

    let entities = (0..ARCHETYPE_PAGE_ROWS)
        .map(|index| {
            world
                .spawn_with_storage(
                    ArchetypeStorageKind::Fragmented,
                    (Position {
                        x: index as i32,
                        y: 0,
                    },),
                )
                .unwrap()
        })
        .collect::<Vec<_>>();

    for entity in entities.iter().take(8) {
        world.despawn(*entity).unwrap();
    }

    let tracked = entities[10];
    let tracked_before = world.entities.location(tracked).unwrap();
    let archetype = world
        .components()
        .get_archetype(tracked_before.archetype_id)
        .unwrap();
    let stats = archetype.storage_stats(ArchetypeStorageKind::Fragmented);

    assert_eq!(stats.live_rows, ARCHETYPE_PAGE_ROWS - 8);
    assert_eq!(stats.dead_rows, 8);
    assert_eq!(stats.dead_ratio_percent(), 12);

    let report = world.compact_fragmented_archetypes(CompactionPolicy::new(ARCHETYPE_PAGE_ROWS, 0));

    assert_eq!(report.compacted_archetypes, 0);
    assert_eq!(report.moved_rows, 0);
    assert_eq!(world.entities.location(tracked), Some(tracked_before));

    let work_version_before_compaction = world.planning().query_work_version();

    let report = world.compact_fragmented_archetypes(CompactionPolicy::aggressive());

    assert_eq!(report.compacted_archetypes, 1);
    assert!(report.moved_rows > 0);
    assert_ne!(
        world.planning().query_work_version(),
        work_version_before_compaction
    );
    assert_eq!(world.entities.location(tracked).unwrap().row, 2);

    let archetype = world
        .components()
        .get_archetype(tracked_before.archetype_id)
        .unwrap();
    let stats = archetype.storage_stats(ArchetypeStorageKind::Fragmented);

    assert_eq!(stats.live_rows, ARCHETYPE_PAGE_ROWS - 8);
    assert_eq!(stats.dead_rows, 0);
}

#[test]
fn query_returns_registered_sparse_components() {
    let mut world = World::new();
    register_components(&mut world);

    world.spawn((Health(5),)).unwrap();
    world.spawn((Position { x: 1, y: 0 }, Health(7))).unwrap();
    world.spawn((Position { x: 2, y: 0 },)).unwrap();

    let mut query = world.query::<&Health, ()>().unwrap();
    let values: Vec<_> = query.iter().map(|health| health.0).collect();

    assert_eq!(sorted(values), vec![5, 7]);
}

#[test]
fn query_mut_can_read_and_write_different_components() {
    let mut world = World::new();
    register_components(&mut world);

    let first = world
        .spawn((Position { x: 1, y: 2 }, Velocity { x: 10, y: 20 }))
        .unwrap();
    let second = world
        .spawn((Position { x: 3, y: 4 }, Velocity { x: 30, y: 40 }))
        .unwrap();

    {
        let mut query = world.query_mut::<(&mut Position, &Velocity), ()>().unwrap();
        for (mut position, velocity) in query.iter_mut() {
            position.x += velocity.x;
            position.y += velocity.y;
        }
    }

    assert_eq!(
        world.get::<Position>(first),
        Some(&Position { x: 11, y: 22 })
    );
    assert_eq!(
        world.get::<Position>(second),
        Some(&Position { x: 33, y: 44 })
    );
}

#[test]
fn query_can_include_entity_ids() {
    let mut world = World::new();
    register_components(&mut world);

    let first = world.spawn((Position { x: 10, y: 0 },)).unwrap();
    let second = world.spawn((Position { x: 20, y: 0 },)).unwrap();

    let mut query = world.query::<(Entity, &Position), ()>().unwrap();
    let mut values: Vec<_> = query.iter().map(|(entity, pos)| (pos.x, entity)).collect();
    values.sort_unstable_by_key(|(x, _)| *x);

    assert_eq!(values, vec![(10, first), (20, second)]);
}

#[test]
fn optional_table_query_reports_present_and_missing_components() {
    let mut world = World::new();
    register_components(&mut world);

    world
        .spawn((Position { x: 1, y: 0 }, Name("Alice".to_owned())))
        .unwrap();
    world.spawn((Position { x: 2, y: 0 },)).unwrap();

    let mut query = world.query::<(&Position, Option<&Name>), ()>().unwrap();
    let mut values: Vec<_> = query
        .iter()
        .map(|(position, name)| (position.x, name.map(|name| name.0.clone())))
        .collect();
    values.sort_unstable_by_key(|(x, _)| *x);

    assert_eq!(values, vec![(1, Some("Alice".to_owned())), (2, None)]);
}

#[test]
fn optional_sparse_query_reports_present_and_missing_components() {
    let mut world = World::new();
    register_components(&mut world);

    world.spawn((Position { x: 1, y: 0 }, Health(10))).unwrap();
    world.spawn((Position { x: 2, y: 0 },)).unwrap();

    let mut query = world.query::<(&Position, Option<&Health>), ()>().unwrap();
    let mut values: Vec<_> = query
        .iter()
        .map(|(position, health)| (position.x, health.map(|health| health.0)))
        .collect();
    values.sort_unstable_by_key(|(x, _)| *x);

    assert_eq!(values, vec![(1, Some(10)), (2, None)]);
}

#[test]
fn added_filter_matches_recent_table_and_sparse_components() {
    let mut world = World::new();
    register_components(&mut world);

    let table_entity = world.spawn((Position { x: 1, y: 0 },)).unwrap();
    let sparse_entity = world.spawn((Health(10),)).unwrap();

    let mut table_query = world.query::<Entity, Added<Position>>().unwrap();
    let table_matches: Vec<_> = table_query.iter().collect();

    let mut sparse_query = world.query::<Entity, Added<Health>>().unwrap();
    let sparse_matches: Vec<_> = sparse_query.iter().collect();

    assert_eq!(table_matches, vec![table_entity]);
    assert_eq!(sparse_matches, vec![sparse_entity]);
}

#[test]
fn get_mut_writes_stay_visible_to_changed_queries() {
    let mut world = World::new();
    register_components(&mut world);

    let entity = world.spawn((Position { x: 1, y: 0 },)).unwrap();

    // Consume the spawn marks so the next query window starts clean.
    let mut state = world.query_state::<&Position, Changed<Position>>().unwrap();
    {
        let mut query = world.query_with(&mut state).unwrap();
        let mut seen = 0;
        query.for_each(|_| seen += 1);
        assert_eq!(seen, 1);
    }

    world.get_mut::<Position>(entity).unwrap().x += 1;

    let mut query = world.query_with(&mut state).unwrap();
    let mut seen = 0;
    query.for_each(|_| seen += 1);
    assert_eq!(seen, 1, "get_mut marks must reach Changed queries");
}

#[test]
fn optional_mut_query_updates_only_existing_components() {
    let mut world = World::new();
    register_components(&mut world);

    let with_mana = world.spawn((Position { x: 1, y: 0 }, Mana(15))).unwrap();
    let without_mana = world.spawn((Position { x: 2, y: 0 },)).unwrap();

    {
        let mut query = world.query_mut::<Option<&mut Mana>, ()>().unwrap();
        for mut mana in query.iter_mut().flatten() {
            mana.0 += 5;
        }
    }

    assert_eq!(world.get::<Mana>(with_mana), Some(&Mana(20)));
    assert_eq!(world.get::<Mana>(without_mana), None);
}

#[test]
fn query_table_filters_select_matching_archetypes() {
    let mut world = World::new();
    register_components(&mut world);

    world.spawn((Position { x: 1, y: 0 }, Player)).unwrap();
    world.spawn((Position { x: 2, y: 0 }, Enemy)).unwrap();
    world.spawn((Position { x: 3, y: 0 },)).unwrap();

    let mut players = world.query::<&Position, With<Player>>().unwrap();
    assert_eq!(
        players
            .iter()
            .map(|position| position.x)
            .collect::<Vec<_>>(),
        vec![1]
    );

    let mut non_enemies = world.query::<&Position, Without<Enemy>>().unwrap();
    let values: Vec<_> = non_enemies.iter().map(|position| position.x).collect();
    assert_eq!(sorted(values), vec![1, 3]);
}

#[test]
fn query_sparse_filters_are_checked_per_entity() {
    let mut world = World::new();
    register_components(&mut world);

    world.spawn((Position { x: 1, y: 0 }, Health(10))).unwrap();
    world.spawn((Position { x: 2, y: 0 },)).unwrap();
    world.spawn((Position { x: 3, y: 0 }, Health(30))).unwrap();

    let mut with_health = world.query::<&Position, With<Health>>().unwrap();
    let values: Vec<_> = with_health.iter().map(|position| position.x).collect();
    assert_eq!(sorted(values), vec![1, 3]);

    let mut without_health = world.query::<&Position, Without<Health>>().unwrap();
    assert_eq!(
        without_health
            .iter()
            .map(|position| position.x)
            .collect::<Vec<_>>(),
        vec![2]
    );
}

#[test]
fn tuple_filters_combine_their_conditions() {
    let mut world = World::new();
    register_components(&mut world);

    world.spawn((Position { x: 1, y: 0 }, Player)).unwrap();
    world
        .spawn((Position { x: 2, y: 0 }, Player, Enemy))
        .unwrap();
    world.spawn((Position { x: 3, y: 0 }, Enemy)).unwrap();

    let mut query = world
        .query::<&Position, (With<Player>, Without<Enemy>)>()
        .unwrap();

    assert_eq!(
        query.iter().map(|position| position.x).collect::<Vec<_>>(),
        vec![1]
    );
}

#[test]
fn or_filter_matches_any_table_component() {
    let mut world = World::new();
    register_components(&mut world);

    world.spawn((Position { x: 1, y: 0 }, Player)).unwrap();
    world.spawn((Position { x: 2, y: 0 }, Enemy)).unwrap();
    world
        .spawn((Position { x: 3, y: 0 }, Player, Enemy))
        .unwrap();
    world.spawn((Position { x: 4, y: 0 },)).unwrap();

    let mut query = world
        .query::<&Position, Or<(With<Player>, With<Enemy>)>>()
        .unwrap();
    let values: Vec<_> = query.iter().map(|position| position.x).collect();

    assert_eq!(sorted(values), vec![1, 2, 3]);
}

#[test]
fn any_of_filter_matches_any_sparse_component() {
    let mut world = World::new();
    register_components(&mut world);

    world.spawn((Position { x: 1, y: 0 }, Health(10))).unwrap();
    world.spawn((Position { x: 2, y: 0 }, Frozen)).unwrap();
    world
        .spawn((Position { x: 3, y: 0 }, Health(30), Frozen))
        .unwrap();
    world.spawn((Position { x: 4, y: 0 },)).unwrap();

    let mut query = world
        .query::<&Position, AnyOf<(With<Health>, With<Frozen>)>>()
        .unwrap();
    let values: Vec<_> = query.iter().map(|position| position.x).collect();

    assert_eq!(sorted(values), vec![1, 2, 3]);
}

#[test]
fn not_filter_inverts_table_component_presence() {
    let mut world = World::new();
    register_components(&mut world);

    world.spawn((Position { x: 1, y: 0 }, Player)).unwrap();
    world.spawn((Position { x: 2, y: 0 }, Enemy)).unwrap();
    world.spawn((Position { x: 3, y: 0 },)).unwrap();

    let mut query = world.query::<&Position, Not<With<Player>>>().unwrap();
    let values: Vec<_> = query.iter().map(|position| position.x).collect();

    assert_eq!(sorted(values), vec![2, 3]);
}

#[test]
fn not_filter_inverts_sparse_component_presence() {
    let mut world = World::new();
    register_components(&mut world);

    world.spawn((Position { x: 1, y: 0 }, Health(10))).unwrap();
    world.spawn((Position { x: 2, y: 0 },)).unwrap();
    world.spawn((Position { x: 3, y: 0 }, Health(30))).unwrap();

    let mut query = world.query::<&Position, Not<With<Health>>>().unwrap();
    let values: Vec<_> = query.iter().map(|position| position.x).collect();

    assert_eq!(values, vec![2]);
}

#[test]
fn not_filter_composes_with_or() {
    let mut world = World::new();
    register_components(&mut world);

    world.spawn((Position { x: 1, y: 0 }, Player)).unwrap();
    world.spawn((Position { x: 2, y: 0 }, Enemy)).unwrap();
    world
        .spawn((Position { x: 3, y: 0 }, Player, Enemy))
        .unwrap();
    world.spawn((Position { x: 4, y: 0 },)).unwrap();

    let mut query = world
        .query::<&Position, Not<Or<(With<Player>, With<Enemy>)>>>()
        .unwrap();
    let values: Vec<_> = query.iter().map(|position| position.x).collect();

    assert_eq!(values, vec![4]);
}

#[test]
fn query_all_is_the_unfiltered_read_query_shortcut() {
    let mut world = World::new();
    register_components(&mut world);

    world.spawn((Position { x: 1, y: 0 },)).unwrap();
    world.spawn((Position { x: 2, y: 0 },)).unwrap();

    let mut query = world.query_all::<&Position>().unwrap();
    let values: Vec<_> = query.iter().map(|position| position.x).collect();

    assert_eq!(sorted(values), vec![1, 2]);
}

#[test]
fn query_all_mut_is_the_unfiltered_write_query_shortcut() {
    let mut world = World::new();
    register_components(&mut world);

    let entity = world.spawn((Position { x: 1, y: 2 },)).unwrap();

    {
        let mut query = world.query_all_mut::<&mut Position>().unwrap();
        for mut position in query.iter_mut() {
            position.y += 10;
        }
    }

    assert_eq!(
        world.get::<Position>(entity),
        Some(&Position { x: 1, y: 12 })
    );
}

#[test]
fn query_each_runs_a_read_callback_for_every_match() {
    let mut world = World::new();
    register_components(&mut world);

    world.spawn((Position { x: 4, y: 0 }, Player)).unwrap();
    world.spawn((Position { x: 5, y: 0 },)).unwrap();

    let mut values = Vec::new();
    world
        .query_each::<&Position, With<Player>, _>(|position| values.push(position.x))
        .unwrap();

    assert_eq!(values, vec![4]);
}

#[test]
fn query_each_mut_runs_a_write_callback_for_every_match() {
    let mut world = World::new();
    register_components(&mut world);

    let first = world.spawn((Position { x: 1, y: 1 }, Player)).unwrap();
    let second = world.spawn((Position { x: 2, y: 2 },)).unwrap();

    world
        .query_each_mut::<&mut Position, With<Player>, _>(|mut position| position.x *= 10)
        .unwrap();

    assert_eq!(
        world.get::<Position>(first),
        Some(&Position { x: 10, y: 1 })
    );
    assert_eq!(
        world.get::<Position>(second),
        Some(&Position { x: 2, y: 2 })
    );
}

#[test]
fn query_state_reuse_sees_archetypes_created_later() {
    let mut world = World::new();
    register_components(&mut world);

    let mut state = world.query_state::<&Position, ()>().unwrap();

    assert_eq!(world.query_with(&mut state).unwrap().iter().count(), 0);

    world.spawn((Position { x: 1, y: 2 },)).unwrap();

    assert_eq!(world.query_with(&mut state).unwrap().iter().count(), 1);
}

#[test]
fn query_mut_with_reuses_state_and_can_update_new_archetypes() {
    let mut world = World::new();
    register_components(&mut world);

    let mut state = world.query_state::<&mut Position, ()>().unwrap();
    let entity = world
        .spawn((Position { x: 1, y: 0 }, Velocity { x: 2, y: 0 }))
        .unwrap();

    {
        let mut query = world.query_mut_with(&mut state).unwrap();
        for mut position in query.iter_mut() {
            position.x += 1;
        }
    }

    assert_eq!(
        world.get::<Position>(entity),
        Some(&Position { x: 2, y: 0 })
    );
}

#[test]
fn query_state_reuse_reflects_removed_rows() {
    let mut world = World::new();
    register_components(&mut world);

    let first = world.spawn((Position { x: 1, y: 0 },)).unwrap();
    let second = world.spawn((Position { x: 2, y: 0 },)).unwrap();
    let mut state = world.query_state::<&Position, ()>().unwrap();

    assert_eq!(world.query_with(&mut state).unwrap().iter().count(), 2);
    world.despawn(first).unwrap();

    let mut values: Vec<_> = world
        .query_with(&mut state)
        .unwrap()
        .iter()
        .map(|position| position.x)
        .collect();
    values.sort_unstable();

    assert_eq!(values, vec![2]);
    assert!(world.contains(second));
}

#[test]
fn query_unregistered_component_returns_query_error() {
    let world = World::new();

    let result = world.query::<&Unregistered, ()>();

    assert!(matches!(
        result,
        Err(WorldError::Query(QueryError::ComponentNotRegistered(_)))
    ));
}

#[test]
fn mutable_query_rejects_read_write_conflict() {
    let mut world = World::new();
    register_components(&mut world);

    let result = world.query_mut::<(&Position, &mut Position), ()>();

    assert!(matches!(
        result,
        Err(WorldError::Query(QueryError::AccessConflict { .. }))
    ));
}

#[test]
fn mutable_query_rejects_duplicate_write_conflict() {
    let mut world = World::new();
    register_components(&mut world);

    let result = world.query_mut::<(&mut Position, &mut Position), ()>();

    assert!(matches!(
        result,
        Err(WorldError::Query(QueryError::AccessConflict { .. }))
    ));
}

#[test]
fn mutable_query_rejects_optional_write_conflict() {
    let mut world = World::new();
    register_components(&mut world);

    let result = world.query_mut::<(&Position, Option<&mut Position>), ()>();

    assert!(matches!(
        result,
        Err(WorldError::Query(QueryError::AccessConflict { .. }))
    ));
}

#[test]
fn entity_ref_exposes_id_and_components_for_live_entities() {
    let mut world = World::new();
    register_components(&mut world);

    let entity = world.spawn((Position { x: 6, y: 7 }, Health(8))).unwrap();
    let view = world.entity(entity).unwrap();

    assert_eq!(view.id(), entity);
    assert_eq!(view.get::<Position>(), Some(&Position { x: 6, y: 7 }));
    assert_eq!(view.get::<Health>(), Some(&Health(8)));
}

#[test]
fn entity_ref_returns_none_for_dead_entities() {
    let mut world = World::new();

    let entity = world.spawn_empty().unwrap();
    world.despawn(entity).unwrap();

    assert!(world.entity(entity).is_none());
    assert!(world.entity(Entity::INVALID).is_none());
}

#[test]
fn entity_mut_can_read_update_and_insert_components() {
    let mut world = World::new();
    register_components(&mut world);

    let entity = world.spawn((Position { x: 1, y: 2 },)).unwrap();

    {
        let mut view = world.entity_mut(entity).unwrap();
        assert_eq!(view.id(), entity);
        assert_eq!(view.get::<Position>(), Some(&Position { x: 1, y: 2 }));
        view.get_mut::<Position>().unwrap().y = 20;
        view.insert((Health(55),)).unwrap();
    }

    assert_eq!(
        world.get::<Position>(entity),
        Some(&Position { x: 1, y: 20 })
    );
    assert_eq!(world.get::<Health>(entity), Some(&Health(55)));
}

#[test]
fn entity_mut_can_despawn_the_entity_it_wraps() {
    let mut world = World::new();
    register_components(&mut world);

    let entity = world.spawn((Position { x: 1, y: 2 },)).unwrap();

    world.entity_mut(entity).unwrap().despawn().unwrap();

    assert!(!world.contains(entity));
    assert!(world.entity_mut(entity).is_none());
}

#[test]
fn world_can_insert_and_get_resource() {
    struct Time {
        delta: f32,
    }

    let mut world = World::new();

    world.insert_resource(Time { delta: 0.5 });

    assert_eq!(world.get_resource::<Time>().unwrap().delta, 0.5);
}

#[test]
fn insert_resource_returns_previous_value() {
    struct Counter(i32);

    let mut world = World::new();

    assert!(world.insert_resource(Counter(1)).is_none());

    let old = world.insert_resource(Counter(2)).unwrap();

    assert_eq!(old.0, 1);
    assert_eq!(world.get_resource::<Counter>().unwrap().0, 2);
}

#[test]
fn res_param_reads_resource() {
    struct Time {
        delta: f32,
    }

    fn read_time(time: Res<Time>) {
        assert_eq!(time.delta, 0.5);
    }

    let mut world = World::new();
    let mut executor = ScheduleExecutor::default();
    world.insert_resource(Time { delta: 0.5 });

    let mut schedule = Schedule::new();
    schedule.add_system(read_time);

    schedule.run(&mut world, &mut executor).unwrap();
}

#[test]
fn res_mut_param_writes_resource() {
    struct Counter(i32);

    fn increment(mut counter: ResMut<Counter>) {
        counter.0 += 1;
    }

    let mut world = World::new();
    let mut executor = ScheduleExecutor::default();
    world.insert_resource(Counter(0));

    let mut schedule = Schedule::new();
    schedule.add_system(increment);
    schedule.run(&mut world, &mut executor).unwrap();

    assert_eq!(world.get_resource::<Counter>().unwrap().0, 1);
}

#[test]
fn tuple_param_rejects_resource_read_write_conflict() {
    struct Time;

    type Params = (Res<'static, Time>, ResMut<'static, Time>);

    let mut world = World::new();

    let result = <Params as SystemParam>::init_state(&mut world);

    assert!(matches!(result, Err(WorldError::Resource(_))));
}

#[test]
fn tuple_param_allows_multiple_resource_reads() {
    struct Time;

    type Params = (Res<'static, Time>, Res<'static, Time>);

    let mut world = World::new();

    let result = <Params as SystemParam>::init_state(&mut world);

    assert!(result.is_ok());
}

#[test]
fn missing_resource_returns_error_when_running_system() {
    struct Time;

    fn needs_time(_time: Res<Time>) {}

    let mut world = World::new();
    let mut executor = ScheduleExecutor::default();

    let mut schedule = Schedule::new();
    schedule.add_system(needs_time);

    let err = schedule.run(&mut world, &mut executor).unwrap_err();

    assert!(matches!(err, ScheduleError::SystemRun { .. }));
}

#[test]
fn system_can_read_one_resource_and_write_another() {
    struct Time {
        delta: i32,
    }

    struct Counter(i32);

    fn update(time: Res<Time>, mut counter: ResMut<Counter>) {
        counter.0 += time.delta;
    }

    let mut world = World::new();
    let mut executor = ScheduleExecutor::default();
    world.insert_resource(Time { delta: 2 });
    world.insert_resource(Counter(0));

    let mut schedule = Schedule::new();
    schedule.add_system(update);

    schedule.run(&mut world, &mut executor).unwrap();

    assert_eq!(world.get_resource::<Counter>().unwrap().0, 2);
}

#[test]
fn resource_len_tracks_insert_replace_and_remove() {
    struct Time;
    struct Counter;

    let mut world = World::new();

    assert_eq!(world.resource_len(), 0);

    world.insert_resource(Time);
    assert_eq!(world.resource_len(), 1);

    world.insert_resource(Time);
    assert_eq!(world.resource_len(), 1);

    world.insert_resource(Counter);
    assert_eq!(world.resource_len(), 2);

    assert!(world.remove_resource::<Time>().is_some());
    assert_eq!(world.resource_len(), 1);
    assert!(!world.contains_resource::<Time>());
    assert!(world.contains_resource::<Counter>());
}

#[test]
fn get_resource_mut_updates_resource_in_place() {
    struct Counter(i32);

    let mut world = World::new();
    world.insert_resource(Counter(4));

    world.get_resource_mut::<Counter>().unwrap().0 += 6;

    assert_eq!(world.get_resource::<Counter>().unwrap().0, 10);
}

#[test]
fn init_resource_inserts_default_only_when_missing() {
    #[derive(Default)]
    struct Counter(i32);

    let mut world = World::new();

    world.init_resource::<Counter>().0 += 5;
    world.init_resource::<Counter>().0 += 2;

    assert_eq!(world.get_resource::<Counter>().unwrap().0, 7);
    assert_eq!(world.resource_len(), 1);
}

#[test]
fn get_resource_or_insert_with_keeps_existing_resource() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Counter(i32);

    let calls = AtomicUsize::new(0);
    let mut world = World::new();
    world.insert_resource(Counter(11));

    let value = world.get_resource_or_insert_with(|| {
        calls.fetch_add(1, Ordering::Relaxed);
        Counter(99)
    });

    assert_eq!(value.0, 11);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(world.resource_len(), 1);
}

#[test]
fn resource_can_be_inserted_after_schedule_initialize() {
    struct Time {
        delta: i32,
    }

    struct Counter(i32);

    fn update(time: Res<Time>, mut counter: ResMut<Counter>) {
        counter.0 += time.delta;
    }

    let mut world = World::new();
    let mut executor = ScheduleExecutor::default();
    world.insert_resource(Counter(0));

    let mut schedule = Schedule::new();
    schedule.add_system(update);
    schedule.initialize(&mut world).unwrap();

    world.insert_resource(Time { delta: 3 });

    schedule.run(&mut world, &mut executor).unwrap();

    assert_eq!(world.get_resource::<Counter>().unwrap().0, 3);
}

#[test]
fn removed_resource_fails_at_run_after_schedule_initialize() {
    struct Time;

    fn needs_time(_time: Res<Time>) {}

    let mut world = World::new();
    let mut executor = ScheduleExecutor::default();
    world.insert_resource(Time);

    let mut schedule = Schedule::new();
    schedule.add_system(needs_time);
    schedule.initialize(&mut world).unwrap();

    assert!(world.remove_resource::<Time>().is_some());

    let err = schedule.run(&mut world, &mut executor).unwrap_err();

    assert!(matches!(err, ScheduleError::SystemRun { .. }));
}

#[test]
fn query_state_rejects_another_world() {
    let mut first = World::new();
    let mut second = World::new();

    register_components(&mut first);
    register_components(&mut second);

    let mut state = first.query_state::<&Position, ()>().unwrap();
    let result = second.query_with(&mut state);

    assert!(matches!(
        result,
        Err(WorldError::Query(QueryError::WorldMismatch {
            expected,
            actual,
        })) if expected == first.id() && actual == second.id()
    ));
}

#[test]
fn optional_mut_writes_stay_visible_to_changed_queries() {
    let mut world = World::new();
    register_components(&mut world);

    let _ = world.spawn((Position { x: 1, y: 0 }, Mana(15))).unwrap();

    let mut state = world.query_state::<Entity, Changed<Mana>>().unwrap();
    {
        let mut query = world.query_with(&mut state).unwrap();
        let mut seen = 0;
        query.for_each(|_| seen += 1);
        assert_eq!(seen, 1);
    }

    {
        let mut query = world.query_mut::<Option<&mut Mana>, ()>().unwrap();
        for mut mana in query.iter_mut().flatten() {
            mana.0 += 5;
        }
    }

    let mut query = world.query_with(&mut state).unwrap();
    let mut seen = 0;
    query.for_each(|_| seen += 1);
    assert_eq!(seen, 1, "Option<&mut> writes must reach Changed queries");
}

#[test]
fn lifecycle_observation_publishes_spawn_and_despawn() {
    let mut world = World::new();
    register_components(&mut world);
    world.observe_spawned();
    world.observe_despawned();

    let entity = world.spawn((Position { x: 1, y: 0 },)).unwrap();

    let spawned: Vec<_> = world
        .get_resource::<Events<Spawned>>()
        .unwrap()
        .records_from(0)
        .map(|record| record.value.entity)
        .collect();
    assert_eq!(spawned, vec![entity]);

    world.despawn(entity).unwrap();

    let despawned: Vec<_> = world
        .get_resource::<Events<Despawned>>()
        .unwrap()
        .records_from(0)
        .map(|record| record.value.entity)
        .collect();
    assert_eq!(despawned, vec![entity]);
}

#[test]
fn change_observation_reports_mutated_entities_per_cycle() {
    let mut world = World::new();
    register_components(&mut world);
    world.observe_changed::<Position>().unwrap();

    // Nothing exists yet: the first dispatch publishes nothing.
    world.dispatch_changed().unwrap();
    assert_eq!(
        world
            .get_resource::<Events<Mutated<Position>>>()
            .unwrap()
            .len(),
        0
    );

    let entity = world.spawn((Position { x: 1, y: 0 },)).unwrap();

    // The spawn's own write happens after registration, so it is
    // reported as the first change set.
    world.dispatch_changed().unwrap();
    let events = world.get_resource::<Events<Mutated<Position>>>().unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(
        events
            .records_from(0)
            .next()
            .unwrap()
            .value
            .entities
            .to_vec(),
        vec![entity]
    );

    {
        let mut position = world.get_mut::<Position>(entity).unwrap();
        position.x += 1;
    }

    world.dispatch_changed().unwrap();
    assert_eq!(
        world
            .get_resource::<Events<Mutated<Position>>>()
            .unwrap()
            .len(),
        2
    );

    // Idle cycle: nothing changes, nothing is published.
    world.dispatch_changed().unwrap();
    assert_eq!(
        world
            .get_resource::<Events<Mutated<Position>>>()
            .unwrap()
            .len(),
        2
    );
}

#[test]
fn add_child_links_both_sides_and_reparents() {
    let mut world = World::new();
    world.init_hierarchy().unwrap();

    let parent = world.spawn_empty().unwrap();
    let child = world.spawn_empty().unwrap();
    let other = world.spawn_empty().unwrap();

    world.add_child(parent, child).unwrap();
    assert_eq!(world.parent(child), Some(parent));
    assert_eq!(world.children(parent), Some(&[child][..]));

    world.add_child(other, child).unwrap();
    assert_eq!(world.parent(child), Some(other));
    assert_eq!(world.children(other), Some(&[child][..]));
    assert_eq!(world.children(parent), Some(&[][..]));
}

#[test]
fn despawn_detaches_from_parent_and_orphans_children() {
    let mut world = World::new();
    world.init_hierarchy().unwrap();

    let parent = world.spawn_empty().unwrap();
    let child = world.spawn_empty().unwrap();
    let grandchild = world.spawn_empty().unwrap();
    world.add_child(parent, child).unwrap();
    world.add_child(child, grandchild).unwrap();

    world.despawn(child).unwrap();

    assert_eq!(world.children(parent), Some(&[][..]));
    assert!(world.contains(grandchild));
    assert_eq!(world.parent(grandchild), None);
}

#[test]
fn despawn_descendants_clears_the_subtree() {
    let mut world = World::new();
    world.init_hierarchy().unwrap();

    let root = world.spawn_empty().unwrap();
    let child = world.spawn_empty().unwrap();
    let grandchild = world.spawn_empty().unwrap();
    world.add_child(root, child).unwrap();
    world.add_child(child, grandchild).unwrap();

    world.despawn_descendants(root).unwrap();
    assert!(world.contains(root));
    assert!(!world.contains(child));
    assert!(!world.contains(grandchild));

    world.despawn_with_descendants(root).unwrap();
    assert!(!world.contains(root));
}

#[test]
fn run_if_skips_systems_until_the_condition_holds() {
    struct Counter(u32);

    fn increment(mut counter: ResMut<Counter>) {
        counter.0 += 1;
    }

    let running = |world: &World| {
        world
            .get_resource::<Counter>()
            .is_some_and(|counter| counter.0 > 0)
    };

    let mut world = World::new();
    world.insert_resource(Counter(0));

    let mut schedule = Schedule::new();
    schedule.add_system(increment).run_if(running);

    let mut executor = ScheduleExecutor::new(2);
    let mut single = ScheduleExecutor::single_threaded();

    // Condition false: the system is skipped in both executors.
    schedule.run(&mut world, &mut executor).unwrap();
    schedule.run(&mut world, &mut single).unwrap();
    assert_eq!(world.get_resource::<Counter>().unwrap().0, 0);

    // The condition flips: the system runs once per cycle again.
    world.get_resource_mut::<Counter>().unwrap().0 = 5;

    schedule.run(&mut world, &mut single).unwrap();
    assert_eq!(world.get_resource::<Counter>().unwrap().0, 6);

    schedule.run(&mut world, &mut executor).unwrap();
    assert_eq!(world.get_resource::<Counter>().unwrap().0, 7);
}

#[test]
fn add_child_rejects_cycles_and_unknown_entities() {
    let mut world = World::new();
    world.init_hierarchy().unwrap();

    let parent = world.spawn_empty().unwrap();
    let child = world.spawn_empty().unwrap();
    world.add_child(parent, child).unwrap();

    assert!(world.add_child(child, parent).is_err());
    assert!(world.add_child(parent, parent).is_err());
    assert!(world.add_child(Entity::INVALID, parent).is_err());
    assert!(world.add_child(parent, Entity::INVALID).is_err());
}
