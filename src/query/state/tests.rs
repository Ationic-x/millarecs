//! Tests for query-state caching, cached work reuse, and route selection.

use millarecs_macros::Component;

use crate::{
    entity::Entity,
    query::{
        filter::{Added, With},
        planning::{QueryRoute, SparseRouteReason},
    },
    world::World,
};

use super::*;

#[derive(Default, Component, Debug, Clone, Copy, PartialEq, Eq)]
struct Position(i32);

#[derive(Default, Component, Debug, PartialEq, Eq)]
struct Marker;

#[derive(Default, Component, Debug, Clone, Copy, PartialEq, Eq)]
struct Health(i32);

#[derive(Default, Component, Debug, Clone, Copy, PartialEq, Eq)]
struct Mana(i32);

fn setup_world() -> World {
    let mut world = World::new();
    world.register_table::<Position>().unwrap();
    world.register_table::<Marker>().unwrap();
    world.register_sparse::<Health>().unwrap();
    world.register_sparse::<Mana>().unwrap();
    world
}

#[test]
fn cached_query_work_is_reused_until_structure_changes() {
    let mut world = setup_world();
    let first = world.spawn((Position(1),)).unwrap();
    let mut state = world.query_state_cached::<&Position, ()>().unwrap();

    let values = {
        let mut query = world.query_with(&mut state).unwrap();
        query.iter().copied().collect::<Vec<_>>()
    };
    assert_eq!(values, vec![Position(1)]);

    let (initial_key, initial_units) = match &state {
        QueryState::Cached(cached) => {
            let entry = cached
                .work_cache
                .sequential
                .as_ref()
                .expect("the first cached query must build sequential work");
            (entry.key, entry.work.units().as_ptr())
        }
        _ => panic!("query_state_cached must create cached storage"),
    };

    world.get_mut::<Position>(first).unwrap().0 = 2;

    let values = {
        let mut query = world.query_with(&mut state).unwrap();
        query.iter().copied().collect::<Vec<_>>()
    };
    assert_eq!(values, vec![Position(2)]);

    let reused_units = match &state {
        QueryState::Cached(cached) => {
            let entry = cached.work_cache.sequential.as_ref().unwrap();
            assert_eq!(entry.key, initial_key);
            entry.work.units().as_ptr()
        }
        _ => unreachable!(),
    };
    assert_eq!(reused_units, initial_units);

    world.spawn((Position(3),)).unwrap();

    let values = {
        let mut query = world.query_with(&mut state).unwrap();
        query.iter().copied().collect::<Vec<_>>()
    };
    assert_eq!(values, vec![Position(2), Position(3)]);

    match &state {
        QueryState::Cached(cached) => {
            let entry = cached.work_cache.sequential.as_ref().unwrap();
            assert_ne!(entry.key.structure_version, initial_key.structure_version);
            assert_eq!(entry.work.units().as_ptr(), initial_units);
        }
        _ => unreachable!(),
    }
}

#[test]
fn cached_sparse_work_rebuilds_when_membership_changes_at_equal_cardinality() {
    let mut world = setup_world();
    let replacement = world.spawn((Position(1),)).unwrap();

    for value in 2..8 {
        world.spawn((Position(value),)).unwrap();
    }

    let original = world.spawn((Position(8), Health(80))).unwrap();
    let mut state = world
        .query_state_cached::<(Entity, &Position), With<Health>>()
        .unwrap();

    let initial = {
        let mut query = world.query_with(&mut state).unwrap();
        query
            .iter()
            .map(|(entity, position)| (entity, *position))
            .collect::<Vec<_>>()
    };
    assert_eq!(initial, vec![(original, Position(8))]);

    let (initial_version, initial_units) = match &state {
        QueryState::Cached(cached) => {
            let entry = cached.work_cache.sequential.as_ref().unwrap();
            assert!(matches!(entry.key.route, QueryRouteKey::SparseDriver(_)));
            (entry.key.structure_version, entry.work.units().as_ptr())
        }
        _ => unreachable!(),
    };

    world.remove::<Health>(original).unwrap();
    world.insert(replacement, (Health(10),)).unwrap();

    let updated = {
        let mut query = world.query_with(&mut state).unwrap();
        query
            .iter()
            .map(|(entity, position)| (entity, *position))
            .collect::<Vec<_>>()
    };
    assert_eq!(updated, vec![(replacement, Position(1))]);

    match &state {
        QueryState::Cached(cached) => {
            let entry = cached.work_cache.sequential.as_ref().unwrap();
            assert_ne!(entry.key.structure_version, initial_version);
            assert_eq!(entry.work.units().as_ptr(), initial_units);
        }
        _ => unreachable!(),
    }
}

#[test]
fn cached_query_keeps_separate_sequential_and_chunked_work() {
    let mut world = setup_world();
    world.spawn((Position(1),)).unwrap();

    let mut state =
        CachedQueryState::<&Position, ()>::new(world.id(), world.components(), world.planning())
            .unwrap();

    let sequential_units = {
        let (_, work) = state.prepare_work(
            world.components(),
            world.planning(),
            QueryWorkConfig::sequential(),
        );
        work.units().as_ptr()
    };

    let chunked = QueryWorkConfig::chunked(1, 1);
    let chunked_units = {
        let (_, work) = state.prepare_work(world.components(), world.planning(), chunked);
        work.units().as_ptr()
    };

    assert_ne!(sequential_units, chunked_units);
    assert!(state.work_cache.sequential.is_some());
    assert!(state.work_cache.chunked.is_some());

    let reused_chunked_units = {
        let (_, work) = state.prepare_work(world.components(), world.planning(), chunked);
        work.units().as_ptr()
    };
    assert_eq!(reused_chunked_units, chunked_units);
}

#[test]
fn auto_query_promotes_by_archetype_threshold() {
    let mut world = setup_world();
    world.spawn((Position(1),)).unwrap();

    let mut state = AutoQueryState::<&Position, ()>::new(
        world.id(),
        world.components(),
        world.planning(),
        AutoQueryPolicy::new(u32::MAX, 0),
    )
    .unwrap();

    assert!(matches!(state.storage(), QueryStorage::Uncached(_)));

    let values: Vec<_> = state
        .iter(world.components(), world.planning(), world.entities())
        .copied()
        .collect();

    assert_eq!(values, vec![Position(1)]);
    assert!(matches!(state.storage(), QueryStorage::Cached(_)));
}

#[test]
fn uncached_query_scans_archetypes_created_after_initialization() {
    let mut world = setup_world();
    let mut state = UncachedQueryState::<&Position, With<Marker>>::new(
        world.id(),
        world.components(),
        world.planning(),
    )
    .unwrap();

    world.spawn((Position(1),)).unwrap();
    world.spawn((Position(2), Marker)).unwrap();

    let values: Vec<_> = state
        .iter(world.components(), world.planning(), world.entities())
        .copied()
        .collect();

    assert_eq!(values, vec![Position(2)]);
}

#[test]
fn route_uses_sparse_with_as_driver_when_it_is_more_selective() {
    let mut world = setup_world();
    for index in 0..8 {
        world.spawn((Position(index),)).unwrap();
    }
    let selected = world.spawn((Position(99), Health(10))).unwrap();

    let mut state = QueryState::<(Entity, &Position), With<Health>>::new(
        world.id(),
        world.components(),
        world.planning(),
    )
    .unwrap();

    let health = world.components().id_of::<Health>().unwrap();

    assert_eq!(state.route().driver(), Some(health));
    assert_eq!(
        state.route().sparse_reason(),
        Some(SparseRouteReason::RequiredSparseWith)
    );
    assert_eq!(state.route().estimated_rows(), 1);
    assert!(state.route().estimated_archetype_rows() > state.route().estimated_rows());

    let mut query = world.query_with(&mut state).unwrap();
    let matches: Vec<_> = query
        .iter()
        .map(|(entity, position)| (entity, *position))
        .collect();
    assert_eq!(matches, vec![(selected, Position(99))]);
}

#[test]
fn route_is_refreshed_when_sparse_cardinality_changes() {
    let mut world = setup_world();
    let first = world.spawn((Position(1), Health(10))).unwrap();
    let second = world.spawn((Position(2), Health(20))).unwrap();

    let mut state = QueryState::<&Position, With<Health>>::new(
        world.id(),
        world.components(),
        world.planning(),
    )
    .unwrap();
    let health = world.components().id_of::<Health>().unwrap();

    assert!(matches!(state.route(), QueryRoute::Archetype { .. }));

    world.remove::<Health>(first).unwrap();

    let values = {
        let mut query = world.query_with(&mut state).unwrap();
        query.iter().copied().collect::<Vec<_>>()
    };

    assert_eq!(values, vec![Position(2)]);
    assert_eq!(state.route().driver(), Some(health));
    assert_eq!(state.route().estimated_rows(), 1);

    world.remove::<Health>(second).unwrap();

    let values = {
        let mut query = world.query_with(&mut state).unwrap();
        query.iter().copied().collect::<Vec<_>>()
    };

    assert!(values.is_empty());
    assert_eq!(state.route().driver(), Some(health));
    assert_eq!(state.route().estimated_rows(), 0);
}

#[test]
fn route_is_refreshed_when_matched_archetype_rows_change() {
    let mut world = setup_world();
    let mut removed = Vec::new();

    for index in 0..8 {
        removed.push(world.spawn((Position(index),)).unwrap());
    }

    let selected = world.spawn((Position(99), Health(10))).unwrap();

    let mut state = QueryState::<(Entity, &Position), With<Health>>::new(
        world.id(),
        world.components(),
        world.planning(),
    )
    .unwrap();

    let health = world.components().id_of::<Health>().unwrap();
    assert_eq!(state.route().driver(), Some(health));
    assert_eq!(state.route().estimated_rows(), 1);
    assert_eq!(state.route().estimated_archetype_rows(), 9);

    for entity in removed {
        world.despawn(entity).unwrap();
    }

    let matches = {
        let mut query = world.query_with(&mut state).unwrap();
        query
            .iter()
            .map(|(entity, position)| (entity, *position))
            .collect::<Vec<_>>()
    };

    assert_eq!(matches, vec![(selected, Position(99))]);
    assert!(matches!(
        state.route(),
        QueryRoute::Archetype { estimated_rows: 1 }
    ));
}

#[test]
fn route_chooses_the_smallest_required_sparse_driver() {
    let mut world = setup_world();
    for index in 0..5 {
        world.spawn((Health(index),)).unwrap();
    }
    let selected = world.spawn((Health(10), Mana(20))).unwrap();

    let mut state = QueryState::<(Entity, &Health, &Mana), ()>::new(
        world.id(),
        world.components(),
        world.planning(),
    )
    .unwrap();

    let mana = world.components().id_of::<Mana>().unwrap();

    assert_eq!(state.route().driver(), Some(mana));
    assert_eq!(
        state.route().sparse_reason(),
        Some(SparseRouteReason::RequiredSparseData)
    );

    let mut query = world.query_with(&mut state).unwrap();
    let matches: Vec<_> = query
        .iter()
        .map(|(entity, health, mana)| (entity, *health, *mana))
        .collect();
    assert_eq!(matches, vec![(selected, Health(10), Mana(20))]);
}

#[test]
fn sparse_driver_dense_position_survives_swap_remove() {
    let mut world = setup_world();
    for index in 0..6 {
        world.spawn((Health(index),)).unwrap();
    }

    let removed = world.spawn((Health(10), Mana(100))).unwrap();
    let retained = world.spawn((Health(20), Mana(200))).unwrap();
    let mut state = QueryState::<(Entity, &Health, &Mana), ()>::new(
        world.id(),
        world.components(),
        world.planning(),
    )
    .unwrap();

    let mana = world.components().id_of::<Mana>().unwrap();
    assert_eq!(state.route().driver(), Some(mana));

    world.despawn(removed).unwrap();

    let matches: Vec<_> = world
        .query_with(&mut state)
        .unwrap()
        .iter()
        .map(|(entity, health, mana)| (entity, *health, *mana))
        .collect();

    assert_eq!(state.route().driver(), Some(mana));
    assert_eq!(matches, vec![(retained, Health(20), Mana(200))]);
}

#[test]
fn mutable_sparse_driver_dense_position_survives_swap_remove() {
    let mut world = setup_world();
    for index in 0..6 {
        world.spawn((Health(index),)).unwrap();
    }

    let removed = world.spawn((Health(10), Mana(100))).unwrap();
    let retained = world.spawn((Health(20), Mana(200))).unwrap();
    let mut state = QueryState::<(Entity, &Health, &mut Mana), ()>::new(
        world.id(),
        world.components(),
        world.planning(),
    )
    .unwrap();

    let mana = world.components().id_of::<Mana>().unwrap();
    assert_eq!(state.route().driver(), Some(mana));

    world.despawn(removed).unwrap();

    {
        let mut query = world.query_mut_with(&mut state).unwrap();
        let matches: Vec<_> = query
            .iter_mut()
            .map(|(entity, health, mut mana)| {
                mana.0 += 1;
                (entity, *health)
            })
            .collect();
        assert_eq!(matches, vec![(retained, Health(20))]);
    }

    assert_eq!(state.route().driver(), Some(mana));
    assert_eq!(world.get::<Mana>(retained), Some(&Mana(201)));
}

#[test]
fn sparse_added_filter_dense_position_survives_swap_remove() {
    let mut world = setup_world();
    for index in 0..6 {
        world.spawn((Health(index),)).unwrap();
    }

    let removed = world.spawn((Health(10), Mana(100))).unwrap();
    let retained = world.spawn((Health(20), Mana(200))).unwrap();
    let mut state = QueryState::<Entity, (With<Mana>, Added<Mana>)>::new(
        world.id(),
        world.components(),
        world.planning(),
    )
    .unwrap();

    let mana = world.components().id_of::<Mana>().unwrap();
    assert_eq!(state.route().driver(), Some(mana));

    world.despawn(removed).unwrap();

    let matches: Vec<_> = world.query_with(&mut state).unwrap().iter().collect();

    assert_eq!(state.route().driver(), Some(mana));
    assert_eq!(matches, vec![retained]);
}

#[test]
fn optional_sparse_data_uses_matching_filter_driver_position() {
    let mut world = setup_world();
    for index in 0..8 {
        world.spawn((Position(index),)).unwrap();
    }
    let selected = world.spawn((Position(99), Mana(20))).unwrap();

    let mut state = QueryState::<(Entity, &Position, Option<&Mana>), With<Mana>>::new(
        world.id(),
        world.components(),
        world.planning(),
    )
    .unwrap();

    let mana = world.components().id_of::<Mana>().unwrap();
    assert_eq!(state.route().driver(), Some(mana));

    let matches: Vec<_> = world
        .query_with(&mut state)
        .unwrap()
        .iter()
        .map(|(entity, position, mana)| (entity, *position, mana.copied()))
        .collect();

    assert_eq!(matches, vec![(selected, Position(99), Some(Mana(20)))]);
}

#[test]
fn optional_mutable_sparse_data_uses_matching_filter_driver_position() {
    let mut world = setup_world();
    for index in 0..8 {
        world.spawn((Position(index),)).unwrap();
    }
    let selected = world.spawn((Position(99), Mana(20))).unwrap();

    let mut state = QueryState::<(Entity, &Position, Option<&mut Mana>), With<Mana>>::new(
        world.id(),
        world.components(),
        world.planning(),
    )
    .unwrap();

    let mana = world.components().id_of::<Mana>().unwrap();
    assert_eq!(state.route().driver(), Some(mana));

    {
        let mut query = world.query_mut_with(&mut state).unwrap();
        let matches: Vec<_> = query
            .iter_mut()
            .map(|(entity, position, mana)| {
                let mut mana = mana.expect("With<Mana> guarantees the optional value");
                mana.0 += 1;
                (entity, *position)
            })
            .collect();
        assert_eq!(matches, vec![(selected, Position(99))]);
    }

    assert_eq!(state.route().driver(), Some(mana));
    assert_eq!(world.get::<Mana>(selected), Some(&Mana(21)));
}

#[test]
fn route_does_not_use_optional_sparse_data_as_driver() {
    let mut world = setup_world();
    world.spawn((Position(1),)).unwrap();
    world.spawn((Position(2), Health(10))).unwrap();

    let state = QueryState::<(&Position, Option<&Health>), ()>::new(
        world.id(),
        world.components(),
        world.planning(),
    )
    .unwrap();

    assert!(matches!(state.route(), QueryRoute::Archetype { .. }));
    assert_eq!(state.route().driver(), None);
}
