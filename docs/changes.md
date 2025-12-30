Removed emissaries (-12)
Moved player_halo to player | head
Removed player_enchantment (-1)
Removed dungeon_wall_banners (-1)
Removed monster_aberration (-2)
Moved gui to gui|startup
Moved item_armor_back to item_armor_torso
Removed monster_tentacles_*_corners (-16)
Moved monster_tentacles_* to monster_tentacles
Removed dungeon_wall_torches (-5)
Removed player_felids (-5)
Moved gut_spells_conjuration/air/earth/enchantment/fire/ice/necromancy/poison/summoning/translocation/transmutation/abilities/invocations to gui_spells
Removed gui_spells_disciplines
Moved gui_spells_monster/divination to gui_spells_components
Moved all white background sprites in gui_spells to gui_spells_components
Moved item_ring_artefact to item_ring
Removed player_mutations (-6)
Moved gui_commands/tabs/startup to gui
Moved item_armor_* to item_armor
Moved item_book_artefact to item_book
Moved item_amulet_artefact to item_amulet
Moved monster_demons/dragonic/dragons/spriggan/vault/holy to monster
Removed monster_panlord
Moved monster_undead_* to monster_undead
Moved monster_aquatic to monster_animals
Removed monster_unique (-19)
Renamed monster_nonliving to monster_nonstandard
Moved monster_fungi_plants/eyes/amorphous/abyss to monster_nonstandard
Removed monster_statues (-53)
Moved dungeon_trees/vaults/traps to dungeon
Removed dungeon_floor_sigils
Removed dungeon_shops
Moved dungeon_statues to dungeon_altars
Moved dungeon_floor_grass to dungeon_floor
Moved dungeon_wall_abyss to dungeon_wall
Moved dungeon_gateways to dungeon_doors
Moved gui_skills to gui
Renamed item_staff to item_weapon_magic
Moved item_rod_wand to item_weapon_magic
Moved item_misc_runes to item_misc
Renamed item_misc to item
Moved item_gold/food/scroll/potion/food to item
Renamed item_ring to item_accessories
Moved item_amulet to item_accessories
Moved item_weapon_artefact to item_weapon
Moved item_weapon_magic/ranged to item_weapon
Moved misc_brands_bottom_right to misc_icon
Moved misc_*(~blood) to misc
Moved player_hand_right_* to player_hand_right
Moved player_hand_left_misc to player_hand_left
Removed player_transform (-23)
Moved player_hair/draconic_head to player_head
Moved player_draconic_wing to player_body
Removed player_beard (-11)
Moved player_boots/barding to player_legs
Moved player_cloak to player_body
Removed player_gloves (-21)
Refactored dungeon_wall to dungeon_brick/rock/stone
Refactored dungeon_floor to dungeon_manmade/natural
Refactored item_weapon to item_weapon_axe/magic/range/sword
Refactored monster to monster_humanoid
Removed some corner icons from dungeon_water


diff
Only in training_data_og: dungeon|floor|grass
Only in training_data_test: dungeon|floor|manmade
Only in training_data_test: dungeon|floor|natural
Only in training_data_og: dungeon|floor|sigils
Only in training_data_og: dungeon|gateways
Only in training_data_og: dungeon|shops
Only in training_data_og: dungeon|statues
Only in training_data_og: dungeon|traps
Only in training_data_og: dungeon|trees
Only in training_data_og: dungeon|vaults
Only in training_data_og: dungeon|wall|banners
Only in training_data_test: dungeon|wall|brick
Only in training_data_test: dungeon|wall|rock
Only in training_data_test: dungeon|wall|stone
Only in training_data_og: dungeon|wall|torches
Only in training_data_og: emissaries
Only in training_data_og: gui|abilities
Only in training_data_og: gui|commands
Only in training_data_og: gui|invocations
Only in training_data_og: gui|skills
Only in training_data_og: gui|spells|air
Only in training_data_og: gui|spells|conjuration
Only in training_data_og: gui|spells|disciplines
Only in training_data_og: gui|spells|divination
Only in training_data_og: gui|spells|earth
Only in training_data_og: gui|spells|enchantment
Only in training_data_og: gui|spells|fire
Only in training_data_og: gui|spells|ice
Only in training_data_og: gui|spells|monster
Only in training_data_og: gui|spells|necromancy
Only in training_data_og: gui|spells|poison
Only in training_data_og: gui|spells|summoning
Only in training_data_og: gui|spells|translocation
Only in training_data_og: gui|spells|transmutation
Only in training_data_og: gui|startup
Only in training_data_og: gui|tabs
Only in training_data_test: item|accesories
Only in training_data_og: item|amulet
Only in training_data_og: item|amulet|artefact
Only in training_data_og: item|armor|artefact
Only in training_data_og: item|armor|back
Only in training_data_og: item|armor|bardings
Only in training_data_og: item|armor|feet
Only in training_data_og: item|armor|hands
Only in training_data_og: item|armor|headgear
Only in training_data_og: item|armor|shields
Only in training_data_og: item|armor|torso
Only in training_data_og: item|book|artefact
Only in training_data_og: item|food
Only in training_data_og: item|gold
Only in training_data_og: item|misc
Only in training_data_og: item|misc|runes
Only in training_data_og: item|potion
Only in training_data_og: item|ring
Only in training_data_og: item|ring|artefact
Only in training_data_og: item|rod
Only in training_data_og: item|scroll
Only in training_data_og: item|staff
Only in training_data_og: item|wand
Only in training_data_og: item|weapon|artefact
Only in training_data_test: item|weapon|axe
Only in training_data_test: item|weapon|magic
Only in training_data_test: item|weapon|range
Only in training_data_og: item|weapon|ranged
Only in training_data_test: item|weapon|sword
Only in training_data_og: misc|brands
Only in training_data_og: misc|brands|bottom_left
Only in training_data_og: misc|brands|bottom_right
Only in training_data_og: misc|brands|top_left
Only in training_data_og: misc|brands|top_right
Only in training_data_test: misc|icon
Only in training_data_og: misc|numbers
Only in training_data_og: monster|aberration
Only in training_data_og: monster|abyss
Only in training_data_og: monster|amorphous
Only in training_data_og: monster|aquatic
Only in training_data_og: monster|demons
Only in training_data_og: monster|demonspawn
Only in training_data_og: monster|draconic
Only in training_data_og: monster|dragons
Only in training_data_og: monster|eyes
Only in training_data_og: monster|fungi_plants
Only in training_data_og: monster|holy
Only in training_data_test: monster|humanoid
Only in training_data_og: monster|nonliving
Only in training_data_test: monster|nonstandard
Only in training_data_og: monster|panlord
Only in training_data_og: monster|spriggan
Only in training_data_og: monster|statues
Only in training_data_og: monster|tentacles|eldritch_corners
Only in training_data_og: monster|tentacles|eldritch_ends
Only in training_data_og: monster|tentacles|kraken_corners
Only in training_data_og: monster|tentacles|kraken_ends
Only in training_data_og: monster|tentacles|kraken_segments
Only in training_data_og: monster|tentacles|starspawn_corners
Only in training_data_og: monster|tentacles|starspawn_ends
Only in training_data_og: monster|tentacles|starspawn_segments
Only in training_data_og: monster|tentacles|vine_corners
Only in training_data_og: monster|tentacles|vine_ends
Only in training_data_og: monster|tentacles|vine_segments
Only in training_data_og: monster|undead|simulacra
Only in training_data_og: monster|undead|skeletons
Only in training_data_og: monster|undead|spectrals
Only in training_data_og: monster|undead|zombies
Only in training_data_og: monster|unique
Only in training_data_og: monster|vault
Only in training_data_og: player
Only in training_data_og: player|barding
Only in training_data_og: player|beard
Only in training_data_og: player|boots
Only in training_data_og: player|cloak
Only in training_data_og: player|draconic_head
Only in training_data_og: player|draconic_wing
Only in training_data_og: player|enchantment
Only in training_data_og: player|felids
Only in training_data_og: player|gloves
Only in training_data_og: player|hair
Only in training_data_og: player|halo
Only in training_data_og: player|hand_left|misc
Only in training_data_og: player|hand_right|artefact
Only in training_data_og: player|hand_right|misc
Only in training_data_og: player|mutations
Only in training_data_og: player|transform
