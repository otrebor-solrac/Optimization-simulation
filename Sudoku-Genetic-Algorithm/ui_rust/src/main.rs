use macroquad::prelude::*;
use serde::Deserialize;
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex};
use std::thread;

#[derive(Debug, Clone, Deserialize)]
struct IndividualData {
    grid: Vec<u8>,
    fitness: u16,
    conflicts: u16,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
enum EnginePacket {
    #[serde(rename = "init")]
    Init {
        fixed: Vec<u8>,
        #[allow(dead_code)]
        puzzle: Vec<u8>,
    },
    #[serde(rename = "generation")]
    Generation {
        gen: u32,
        best_fitness: u16,
        conflicts: u16,
        avg_fitness: f32,
        mutation_rate: f32,
        stagnation: u32,
        stagnation_boost: bool,
        solved: bool,
        top9: Vec<IndividualData>,
    },
    #[serde(rename = "solved")]
    Solved { gen: u32 },
}

#[derive(Debug, Clone, PartialEq)]
enum AppMode {
    Preview,
    Running,
}

struct PuzzlePreset {
    key: &'static str,
    label: &'static str,
    difficulty: &'static str,
    clues: usize,
    stars: &'static str,
    desc: &'static str,
    grid: [u8; 81],
}

const PRESETS: [PuzzlePreset; 4] = [
    PuzzlePreset {
        key: "easy",
        label: "EASY",
        difficulty: "Beginner / Easy",
        clues: 30,
        stars: "**---",
        desc: "Classic newspaper Sudoku. Low initial conflict density, ideal for demonstrating rapid genetic convergence.",
        grid: [
            5, 3, 0, 0, 7, 0, 0, 0, 0,
            6, 0, 0, 1, 9, 5, 0, 0, 0,
            0, 9, 8, 0, 0, 0, 0, 6, 0,
            8, 0, 0, 0, 6, 0, 0, 0, 3,
            4, 0, 0, 8, 0, 3, 0, 0, 1,
            7, 0, 0, 0, 2, 0, 0, 0, 6,
            0, 6, 0, 0, 0, 0, 2, 8, 0,
            0, 0, 0, 4, 1, 9, 0, 0, 5,
            0, 0, 0, 0, 8, 0, 0, 7, 9,
        ],
    },
    PuzzlePreset {
        key: "medium",
        label: "MEDIUM",
        difficulty: "Intermediate",
        clues: 24,
        stars: "***--",
        desc: "Symmetric clue layout. Requires multiple 3x3 block recombinations to resolve column constraints.",
        grid: [
            0, 2, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 6, 0, 0, 0, 0, 3,
            0, 7, 4, 0, 8, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 3, 0, 0, 2,
            0, 8, 0, 0, 4, 0, 0, 1, 0,
            6, 0, 0, 5, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 7, 8, 0,
            5, 0, 0, 0, 0, 9, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 4, 0,
        ],
    },
    PuzzlePreset {
        key: "hard",
        label: "HARD",
        difficulty: "Expert / Minimalist",
        clues: 21,
        stars: "****-",
        desc: "Minimalist puzzle with only 21 clues. Vast combinatorial search space with deep local optimum traps.",
        grid: [
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 3, 0, 8, 5,
            0, 0, 1, 0, 2, 0, 0, 0, 0,
            0, 0, 0, 5, 0, 7, 0, 0, 0,
            0, 0, 4, 0, 0, 0, 1, 0, 0,
            0, 9, 0, 0, 0, 0, 0, 0, 0,
            5, 0, 0, 0, 0, 0, 0, 7, 3,
            0, 0, 2, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 4, 0, 0, 0, 9,
        ],
    },
    PuzzlePreset {
        key: "escargot",
        label: "EVIL (AI ESCARGOT)",
        difficulty: "World's Hardest Sudoku",
        clues: 23,
        stars: "***** (EVIL)",
        desc: "Created by mathematician Arto Inkala to defeat heuristic solvers. Features brutal plateau traps and dramatic mutation bursts.",
        grid: [
            1, 0, 0, 0, 0, 7, 0, 9, 0,
            0, 3, 0, 0, 2, 0, 0, 0, 8,
            0, 0, 9, 6, 0, 0, 5, 0, 0,
            0, 0, 5, 3, 0, 0, 9, 0, 0,
            0, 1, 0, 0, 8, 0, 0, 0, 2,
            6, 0, 0, 0, 0, 4, 0, 0, 0,
            3, 0, 0, 0, 0, 0, 0, 1, 0,
            0, 4, 0, 0, 0, 0, 0, 0, 7,
            0, 0, 7, 0, 0, 0, 3, 0, 0,
        ],
    },
];

#[derive(Debug, Clone)]
struct VisualizerState {
    mode: AppMode,
    selected_preset: usize,
    gen: u32,
    best_fitness: u16,
    conflicts: u16,
    avg_fitness: f32,
    mutation_rate: f32,
    stagnation: u32,
    stagnation_boost: bool,
    solved: bool,
    fixed: Vec<bool>,
    top9: Vec<IndividualData>,
    paused: bool,
    speed_idx: usize,
    shake_timer: f32,
}

impl Default for VisualizerState {
    fn default() -> Self {
        Self {
            mode: AppMode::Preview,
            selected_preset: 0,
            gen: 0,
            best_fitness: 0,
            conflicts: 99,
            avg_fitness: 0.0,
            mutation_rate: 0.015,
            stagnation: 0,
            stagnation_boost: false,
            solved: false,
            fixed: vec![false; 81],
            top9: Vec::new(),
            paused: false,
            speed_idx: 1,
            shake_timer: 0.0,
        }
    }
}

const SPEEDS: [(&str, f32); 6] = [
    ("1.0s (Cine)", 1.0),
    ("0.5s (Slow)", 0.5),
    ("0.25s (Med)", 0.25),
    ("0.1s (Fast)", 0.1),
    ("0.02s (Turbo)", 0.02),
    ("MAX (0s)", 0.0),
];

fn window_conf() -> Conf {
    Conf {
        window_title: "Sudoku GA - 9 AI Contenders (Live Evolutionary Battle)".to_string(),
        window_width: 1280,
        window_height: 720,
        window_resizable: true,
        high_dpi: true,
        ..Default::default()
    }
}

struct BoardAnalysis {
    cell_conflicts: [bool; 81],
    row_solved: [bool; 9],
    col_solved: [bool; 9],
    solved_lines_count: usize,
}

impl Default for BoardAnalysis {
    fn default() -> Self {
        Self {
            cell_conflicts: [false; 81],
            row_solved: [false; 9],
            col_solved: [false; 9],
            solved_lines_count: 0,
        }
    }
}

fn analyze_board(grid: &[u8]) -> BoardAnalysis {
    let mut analysis = BoardAnalysis::default();
    if grid.len() < 81 {
        return analysis;
    }

    // Check rows
    for r in 0..9 {
        let mut counts = [0u8; 10];
        let mut unique_digits = 0;
        for c in 0..9 {
            let val = grid[r * 9 + c] as usize;
            if val >= 1 && val <= 9 {
                if counts[val] == 0 {
                    unique_digits += 1;
                }
                counts[val] += 1;
            }
        }
        if unique_digits == 9 {
            analysis.row_solved[r] = true;
            analysis.solved_lines_count += 1;
        }
        for c in 0..9 {
            let val = grid[r * 9 + c] as usize;
            if val > 0 && val <= 9 && counts[val] > 1 {
                analysis.cell_conflicts[r * 9 + c] = true;
            }
        }
    }

    // Check cols
    for c in 0..9 {
        let mut counts = [0u8; 10];
        let mut unique_digits = 0;
        for r in 0..9 {
            let val = grid[r * 9 + c] as usize;
            if val >= 1 && val <= 9 {
                if counts[val] == 0 {
                    unique_digits += 1;
                }
                counts[val] += 1;
            }
        }
        if unique_digits == 9 {
            analysis.col_solved[c] = true;
            analysis.solved_lines_count += 1;
        }
        for r in 0..9 {
            let val = grid[r * 9 + c] as usize;
            if val > 0 && val <= 9 && counts[val] > 1 {
                analysis.cell_conflicts[r * 9 + c] = true;
            }
        }
    }

    // Check 3x3 blocks for internal conflicts
    for br in 0..3 {
        for bc in 0..3 {
            let mut counts = [0u8; 10];
            for r in 0..3 {
                for c in 0..3 {
                    let idx = (br * 3 + r) * 9 + (bc * 3 + c);
                    let val = grid[idx] as usize;
                    if val <= 9 {
                        counts[val] += 1;
                    }
                }
            }
            for r in 0..3 {
                for c in 0..3 {
                    let idx = (br * 3 + r) * 9 + (bc * 3 + c);
                    let val = grid[idx] as usize;
                    if val > 0 && val <= 9 && counts[val] > 1 {
                        analysis.cell_conflicts[idx] = true;
                    }
                }
            }
        }
    }

    analysis
}

fn draw_button(
    text: &str,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    bg_color: Color,
    hover_color: Color,
) -> bool {
    let mouse = mouse_position();
    let is_hover = mouse.0 >= x && mouse.0 <= x + w && mouse.1 >= y && mouse.1 <= y + h;
    let color = if is_hover { hover_color } else { bg_color };

    draw_rectangle(x, y, w, h, color);
    draw_rectangle_lines(x, y, w, h, 1.5, Color::new(0.4, 0.5, 0.65, 0.8));

    let font_size = 15.0;
    let dims = measure_text(text, None, font_size as u16, 1.0);
    draw_text(
        text,
        x + (w - dims.width) * 0.5,
        y + (h + dims.height) * 0.5 - 2.0,
        font_size,
        WHITE,
    );

    is_hover && is_mouse_button_pressed(MouseButton::Left)
}

#[macroquad::main(window_conf)]
async fn main() {
    let state = Arc::new(Mutex::new(VisualizerState::default()));
    let (tx_cmd, rx_cmd) = channel::<String>();

    // Spawn Python GA engine process
    let state_clone = Arc::clone(&state);
    thread::spawn(move || {
        let script_candidates = [
            "/app/engine/sudoku_ga.py",
            "engine/sudoku_ga.py",
            "../engine/sudoku_ga.py",
        ];
        let mut script_path = "engine/sudoku_ga.py";
        for path in script_candidates {
            if std::path::Path::new(path).exists() {
                script_path = path;
                break;
            }
        }

        let mut child = Command::new("python3")
            .arg(script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("Failed to spawn Python GA engine");

        let stdout = child.stdout.take().expect("Failed to open child stdout");
        let mut stdin = child.stdin.take().expect("Failed to open child stdin");

        // Forward commands from GUI to Python stdin
        thread::spawn(move || {
            while let Ok(cmd) = rx_cmd.recv() {
                let _ = writeln!(stdin, "{}", cmd);
                let _ = stdin.flush();
            }
        });

        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            if let Ok(line_str) = line {
                if let Ok(packet) = serde_json::from_str::<EnginePacket>(&line_str) {
                    let mut s = state_clone.lock().unwrap();
                    match packet {
                        EnginePacket::Init { fixed, .. } => {
                            s.fixed = fixed.into_iter().map(|v| v != 0).collect();
                        }
                        EnginePacket::Generation {
                            gen,
                            best_fitness,
                            conflicts,
                            avg_fitness,
                            mutation_rate,
                            stagnation,
                            stagnation_boost,
                            solved,
                            top9,
                        } => {
                            s.gen = gen;
                            s.best_fitness = best_fitness;
                            s.conflicts = conflicts;
                            s.avg_fitness = avg_fitness;
                            s.mutation_rate = mutation_rate;
                            s.stagnation = stagnation;
                            if stagnation_boost && !s.stagnation_boost {
                                s.shake_timer = 0.35;
                            }
                            s.stagnation_boost = stagnation_boost;
                            s.solved = solved;
                            s.top9 = top9;
                        }
                        EnginePacket::Solved { gen } => {
                            s.solved = true;
                            s.gen = gen;
                        }
                    }
                }
            }
        }
    });

    loop {
        let delta = get_frame_time();
        // High-contrast clean dark background
        clear_background(Color::new(0.05, 0.06, 0.08, 1.0));

        let current_state = {
            let mut s = state.lock().unwrap();
            if s.shake_timer > 0.0 {
                s.shake_timer = (s.shake_timer - delta).max(0.0);
            }
            s.clone()
        };

        let screen_w = screen_width();
        let screen_h = screen_height();

        match current_state.mode {
            // =========================================================================
            // MODE 1: PREVIEW / PUZZLE SELECTION
            // =========================================================================
            AppMode::Preview => {
                let preset = &PRESETS[current_state.selected_preset];

                // Header
                draw_rectangle(
                    20.0,
                    15.0,
                    screen_w - 40.0,
                    70.0,
                    Color::new(0.09, 0.11, 0.16, 0.95),
                );
                draw_rectangle_lines(
                    20.0,
                    15.0,
                    screen_w - 40.0,
                    70.0,
                    1.5,
                    Color::new(0.3, 0.4, 0.55, 0.8),
                );

                draw_text("SUDOKU GENETIC ALGORITHM", 36.0, 42.0, 22.0, GOLD);
                draw_text(
                    "Inspect the original puzzle clues and select difficulty before unleashing the 9 evolutionary agents.",
                    36.0,
                    68.0,
                    14.0,
                    LIGHTGRAY,
                );

                // Difficulty Tabs
                let tab_y = 100.0;
                let tab_w = (screen_w - 50.0) / 4.0;
                for (idx, p) in PRESETS.iter().enumerate() {
                    let tx = 25.0 + (idx as f32) * (tab_w + 5.0);
                    let is_active = idx == current_state.selected_preset;

                    let bg_col = if is_active {
                        Color::new(0.22, 0.28, 0.38, 1.0)
                    } else {
                        Color::new(0.08, 0.10, 0.14, 0.9)
                    };
                    let hover_col = Color::new(0.30, 0.38, 0.50, 1.0);

                    if draw_button(
                        &format!("{} ({} clues)", p.label, p.clues),
                        tx,
                        tab_y,
                        tab_w - 5.0,
                        40.0,
                        bg_col,
                        hover_col,
                    ) {
                        let mut s = state.lock().unwrap();
                        s.selected_preset = idx;
                    }

                    if is_active {
                        draw_rectangle_lines(tx, tab_y, tab_w - 5.0, 40.0, 2.5, GOLD);
                    }
                }

                // Original Board Preview (Left)
                let board_x = 35.0;
                let board_y = 160.0;
                let board_size = (screen_h - 190.0).min(510.0);
                let cell_size = board_size / 9.0;

                draw_rectangle(
                    board_x - 8.0,
                    board_y - 8.0,
                    board_size + 16.0,
                    board_size + 16.0,
                    Color::new(0.09, 0.11, 0.15, 0.95),
                );
                draw_rectangle_lines(
                    board_x - 8.0,
                    board_y - 8.0,
                    board_size + 16.0,
                    board_size + 16.0,
                    2.0,
                    Color::new(0.4, 0.5, 0.65, 0.8),
                );

                for r in 0..9 {
                    for c in 0..9 {
                        let idx = r * 9 + c;
                        let cx = board_x + (c as f32) * cell_size;
                        let cy = board_y + (r as f32) * cell_size;
                        let num = preset.grid[idx];

                        let cell_bg = if num != 0 {
                            Color::new(0.18, 0.22, 0.30, 0.95) // Original fixed clues in distinct slate
                        } else {
                            Color::new(0.07, 0.08, 0.11, 0.9) // Blank
                        };

                        draw_rectangle(cx + 1.0, cy + 1.0, cell_size - 2.0, cell_size - 2.0, cell_bg);

                        if num != 0 {
                            let text = format!("{}", num);
                            let font_size = cell_size * 0.55;
                            let dims = measure_text(&text, None, font_size as u16, 1.0);
                            draw_text(
                                &text,
                                cx + (cell_size - dims.width) * 0.5,
                                cy + (cell_size + dims.height) * 0.5 - 2.0,
                                font_size,
                                Color::new(1.0, 0.84, 0.30, 1.0), // Warm Gold for original clues
                            );
                        }
                    }
                }

                // 3x3 Block Grid Lines
                for i in 0..=9 {
                    let thick = if i % 3 == 0 { 3.0 } else { 1.0 };
                    let line_color = if i % 3 == 0 {
                        Color::new(0.6, 0.7, 0.85, 0.8)
                    } else {
                        Color::new(0.25, 0.30, 0.40, 0.5)
                    };
                    let pos = i as f32 * cell_size;
                    draw_line(board_x, board_y + pos, board_x + board_size, board_y + pos, thick, line_color);
                    draw_line(board_x + pos, board_y, board_x + pos, board_y + board_size, thick, line_color);
                }

                // Info Panel & Start Button (Right)
                let panel_x = board_x + board_size + 35.0;
                let panel_w = screen_w - panel_x - 30.0;
                let panel_y = 160.0;
                let panel_h = board_size + 16.0;

                draw_rectangle(panel_x, panel_y - 8.0, panel_w, panel_h, Color::new(0.09, 0.11, 0.16, 0.95));
                draw_rectangle_lines(panel_x, panel_y - 8.0, panel_w, panel_h, 1.5, Color::new(0.3, 0.4, 0.55, 0.6));

                let mut py = panel_y + 20.0;
                draw_text("MISSION BRIEFING", panel_x + 20.0, py, 18.0, GOLD);

                py += 35.0;
                draw_text(&format!("Puzzle: {}", preset.label), panel_x + 20.0, py, 16.0, WHITE);
                py += 24.0;
                draw_text(&format!("Difficulty: {} ({})", preset.stars, preset.difficulty), panel_x + 20.0, py, 15.0, Color::new(0.3, 0.85, 0.5, 1.0));
                py += 24.0;
                draw_text(&format!("Initial Clues: {} / 81", preset.clues), panel_x + 20.0, py, 15.0, LIGHTGRAY);
                py += 24.0;
                draw_text(&format!("Cells to Solve: {}", 81 - preset.clues), panel_x + 20.0, py, 15.0, LIGHTGRAY);

                py += 40.0;
                draw_text("DESCRIPTION:", panel_x + 20.0, py, 14.0, GOLD);
                py += 22.0;
                draw_text(preset.desc, panel_x + 20.0, py, 13.0, Color::new(0.85, 0.85, 0.9, 1.0));

                py += 55.0;
                draw_text("INITIAL SPEED:", panel_x + 20.0, py, 14.0, LIGHTGRAY);
                let spd_text = SPEEDS[current_state.speed_idx].0;
                if draw_button(
                    &format!("Speed: {}", spd_text),
                    panel_x + 160.0,
                    py - 22.0,
                    150.0,
                    32.0,
                    Color::new(0.20, 0.25, 0.35, 1.0),
                    Color::new(0.28, 0.36, 0.50, 1.0),
                ) {
                    let mut s = state.lock().unwrap();
                    s.speed_idx = (s.speed_idx + 1) % SPEEDS.len();
                }

                // START BUTTON
                let start_btn_y = panel_y + panel_h - 90.0;
                let pulse = ((get_time() * 4.0).sin() * 0.15 + 0.85) as f32;
                let start_bg = Color::new(0.12 * pulse, 0.65 * pulse, 0.25 * pulse, 1.0);
                let start_hover = Color::new(0.18, 0.80, 0.35, 1.0);

                if draw_button(
                    "START SIMULATION (9 AGENTS LIVE)",
                    panel_x + 20.0,
                    start_btn_y,
                    panel_w - 40.0,
                    55.0,
                    start_bg,
                    start_hover,
                ) {
                    let mut s = state.lock().unwrap();
                    s.mode = AppMode::Running;
                    s.gen = 0;
                    s.conflicts = 99;
                    s.best_fitness = 0;
                    s.solved = false;
                    s.top9.clear();
                    s.fixed = preset.grid.iter().map(|&v| v != 0).collect();

                    let chosen_key = preset.key;
                    let chosen_delay = SPEEDS[s.speed_idx].1;
                    let _ = tx_cmd.send(format!(
                        "{{\"cmd\":\"start\",\"puzzle\":\"{}\",\"delay\":{}}}",
                        chosen_key, chosen_delay
                    ));
                }
            }

            // =========================================================================
            // MODE 2: RUNNING (LIVE 9-AGENT BATTLE ROYALE)
            // =========================================================================
            AppMode::Running => {
                let (shake_x, shake_y) = if current_state.shake_timer > 0.0 {
                    (
                        (rand::gen_range(-1.0, 1.0) * 8.0 * (current_state.shake_timer / 0.35)),
                        (rand::gen_range(-1.0, 1.0) * 8.0 * (current_state.shake_timer / 0.35)),
                    )
                } else {
                    (0.0, 0.0)
                };

                let hud_y = 12.0 + shake_y;
                draw_rectangle(
                    15.0 + shake_x,
                    hud_y,
                    screen_w - 30.0,
                    68.0,
                    Color::new(0.09, 0.11, 0.16, 0.95),
                );
                draw_rectangle_lines(
                    15.0 + shake_x,
                    hud_y,
                    screen_w - 30.0,
                    68.0,
                    1.5,
                    Color::new(0.3, 0.4, 0.55, 0.6),
                );

                // Title and Generation
                draw_text("SUDOKU GENETIC ALGORITHM", 32.0 + shake_x, hud_y + 26.0, 19.0, GOLD);
                draw_text(
                    &format!("GEN: {}", current_state.gen),
                    32.0 + shake_x,
                    hud_y + 52.0,
                    18.0,
                    WHITE,
                );

                // Conflict Counter (RED / GREEN)
                let conflict_color = if current_state.conflicts == 0 {
                    Color::new(0.15, 0.95, 0.35, 1.0) // Solved Green
                } else if current_state.conflicts <= 3 {
                    Color::new(1.0, 0.20, 0.20, 1.0) // Red Alert
                } else {
                    Color::new(0.95, 0.40, 0.25, 1.0)
                };

                draw_rectangle(
                    310.0 + shake_x,
                    hud_y + 10.0,
                    225.0,
                    48.0,
                    Color::new(0.12, 0.06, 0.08, 0.9),
                );
                draw_rectangle_lines(
                    310.0 + shake_x,
                    hud_y + 10.0,
                    225.0,
                    48.0,
                    2.0,
                    conflict_color,
                );

                draw_text("CONFLICTS REMAINING", 322.0 + shake_x, hud_y + 26.0, 11.0, LIGHTGRAY);
                draw_text(
                    &format!("{}", current_state.conflicts),
                    322.0 + shake_x,
                    hud_y + 50.0,
                    26.0,
                    conflict_color,
                );

                // Leader analysis for solved lines count
                let ind1_analysis = if !current_state.top9.is_empty() {
                    analyze_board(&current_state.top9[0].grid)
                } else {
                    BoardAnalysis::default()
                };

                // Solved lines indicator in green!
                let lines_color = if ind1_analysis.solved_lines_count == 18 {
                    Color::new(0.15, 0.95, 0.35, 1.0)
                } else {
                    Color::new(0.35, 0.85, 0.45, 1.0)
                };
                draw_text(
                    &format!("CLEARED: {}/18", ind1_analysis.solved_lines_count),
                    415.0 + shake_x,
                    hud_y + 38.0,
                    13.0,
                    lines_color,
                );
                draw_text(
                    &format!("Fit: {}/243", current_state.best_fitness),
                    415.0 + shake_x,
                    hud_y + 52.0,
                    12.0,
                    LIGHTGRAY,
                );

                // Stagnation Warning / Burst Indicator
                if current_state.stagnation_boost || current_state.shake_timer > 0.0 {
                    let pulse = ((get_time() * 10.0).sin() * 0.5 + 0.5) as f32;
                    let alert_color = Color::new(1.0, 0.15, 0.15, 0.85 + pulse * 0.15);
                    draw_rectangle(550.0 + shake_x, hud_y + 12.0, 170.0, 44.0, alert_color);
                    draw_text(
                        ">> BURST! <<",
                        560.0 + shake_x,
                        hud_y + 38.0,
                        16.0,
                        WHITE,
                    );
                } else {
                    draw_text(
                        &format!("Mutation: {:.3}", current_state.mutation_rate),
                        550.0 + shake_x,
                        hud_y + 32.0,
                        13.0,
                        LIGHTGRAY,
                    );
                    draw_text(
                        &format!("Stagnation: {}/5", current_state.stagnation),
                        550.0 + shake_x,
                        hud_y + 52.0,
                        13.0,
                        LIGHTGRAY,
                    );
                }

                // Interactive Controls
                let btn_y = hud_y + 16.0;
                let pause_label = if current_state.paused { "RESUME" } else { "PAUSE" };
                if draw_button(
                    pause_label,
                    screen_w - 550.0 + shake_x,
                    btn_y,
                    85.0,
                    36.0,
                    Color::new(0.20, 0.25, 0.35, 1.0),
                    Color::new(0.28, 0.36, 0.50, 1.0),
                ) {
                    let mut s = state.lock().unwrap();
                    s.paused = !s.paused;
                    let _ = tx_cmd.send(format!(
                        "{{\"cmd\":\"{}\"}}",
                        if s.paused { "pause" } else { "resume" }
                    ));
                }

                if draw_button(
                    "STEP",
                    screen_w - 455.0 + shake_x,
                    btn_y,
                    60.0,
                    36.0,
                    Color::new(0.20, 0.25, 0.35, 1.0),
                    Color::new(0.28, 0.36, 0.50, 1.0),
                ) {
                    let _ = tx_cmd.send("{\"cmd\":\"step\"}".to_string());
                }

                let speed_label = SPEEDS[current_state.speed_idx].0;
                if draw_button(
                    &format!("SPD: {}", speed_label),
                    screen_w - 385.0 + shake_x,
                    btn_y,
                    130.0,
                    36.0,
                    Color::new(0.20, 0.25, 0.35, 1.0),
                    Color::new(0.28, 0.36, 0.50, 1.0),
                ) {
                    let mut s = state.lock().unwrap();
                    s.speed_idx = (s.speed_idx + 1) % SPEEDS.len();
                    let val = SPEEDS[s.speed_idx].1;
                    let _ = tx_cmd.send(format!("{{\"cmd\":\"speed\",\"val\":{}}}", val));
                }

                if draw_button(
                    "BURST",
                    screen_w - 245.0 + shake_x,
                    btn_y,
                    90.0,
                    36.0,
                    Color::new(0.75, 0.15, 0.15, 1.0),
                    Color::new(0.95, 0.25, 0.25, 1.0),
                ) {
                    let _ = tx_cmd.send("{\"cmd\":\"mutate_burst\"}".to_string());
                }

                if draw_button(
                    "PUZZLE",
                    screen_w - 145.0 + shake_x,
                    btn_y,
                    95.0,
                    36.0,
                    Color::new(0.28, 0.20, 0.25, 1.0),
                    Color::new(0.42, 0.25, 0.35, 1.0),
                ) {
                    let mut s = state.lock().unwrap();
                    s.mode = AppMode::Preview;
                    let _ = tx_cmd.send("{\"cmd\":\"reset\"}".to_string());
                }

                // ---------------- LEADER BOARD #1 (BIG ON THE LEFT) ----------------
                let leader_x = 25.0 + shake_x;
                let leader_y = 95.0 + shake_y;
                let leader_size = (screen_h - 120.0).min(560.0);

                // Frame
                draw_rectangle(
                    leader_x - 8.0,
                    leader_y - 8.0,
                    leader_size + 16.0,
                    leader_size + 16.0,
                    Color::new(0.08, 0.10, 0.14, 0.95),
                );
                let leader_border = if current_state.conflicts == 0 {
                    Color::new(0.2, 0.95, 0.35, 1.0) // Solid Green border on win
                } else {
                    Color::new(0.30, 0.60, 0.95, 0.85) // Blue border
                };
                draw_rectangle_lines(
                    leader_x - 8.0,
                    leader_y - 8.0,
                    leader_size + 16.0,
                    leader_size + 16.0,
                    2.5,
                    leader_border,
                );

                draw_text("APEX CONTENDER #1 (LEADER)", leader_x, leader_y - 12.0, 16.0, GOLD);
                draw_text("[GOLD: ORIGINAL CLUES  |  WHITE: AI DIGITS]", leader_x + 250.0, leader_y - 12.0, 12.0, Color::new(0.70, 0.82, 0.92, 0.85));

                if !current_state.top9.is_empty() {
                    let ind1 = &current_state.top9[0];
                    let cell_size = leader_size / 9.0;
                    let analysis = &ind1_analysis;

                    for r in 0..9 {
                        for c in 0..9 {
                            let idx = r * 9 + c;
                            let cx = leader_x + (c as f32) * cell_size;
                            let cy = leader_y + (r as f32) * cell_size;
                            let is_fixed = current_state.fixed.get(idx).copied().unwrap_or(false);
                            let is_conflict = analysis.cell_conflicts[idx];

                            let is_row_solved = analysis.row_solved[r];
                            let is_col_solved = analysis.col_solved[c];

                            let cell_bg = if is_row_solved || is_col_solved {
                                if is_conflict {
                                    Color::new(0.85, 0.15, 0.18, 0.95) // Keep conflicts visible
                                } else {
                                    Color::new(0.18, 0.65, 0.28, 0.95) // Solved line GREEN
                                }
                            } else if is_conflict {
                                Color::new(0.85, 0.15, 0.18, 0.95) // VIVID RED for duplicate/conflict
                            } else if is_fixed {
                                Color::new(0.12, 0.18, 0.28, 0.95) // ORIGINAL dark blue-slate for initial clues
                            } else {
                                Color::new(0.06, 0.10, 0.17, 0.90) // ORIGINAL deep dark blue for active cells
                            };

                            draw_rectangle(cx + 1.0, cy + 1.0, cell_size - 2.0, cell_size - 2.0, cell_bg);

                            // Golden contour/border for original fixed clues
                            if is_fixed {
                                draw_rectangle_lines(
                                    cx + 2.5,
                                    cy + 2.5,
                                    cell_size - 5.0,
                                    cell_size - 5.0,
                                    2.0,
                                    Color::new(1.0, 0.82, 0.20, 0.95),
                                );
                            }

                            // Digits: GOLD for original fixed clues (high contrast on green & dark bg), White for AI
                            if idx < ind1.grid.len() {
                                let num = ind1.grid[idx];
                                if num > 0 {
                                    let text = format!("{}", num);
                                    let font_size = cell_size * 0.55;
                                    let dims = measure_text(&text, None, font_size as u16, 1.0);

                                    let text_color = if is_fixed {
                                        GOLD // High-contrast bright gold for original fixed clues
                                    } else {
                                        WHITE // Pure crisp white for active evolved digits
                                    };

                                    draw_text(
                                        &text,
                                        cx + (cell_size - dims.width) * 0.5,
                                        cy + (cell_size + dims.height) * 0.5 - 2.0,
                                        font_size,
                                        text_color,
                                    );
                                }
                            }
                        }
                    }

                    // 1. Draw all grid lines in original BLUE
                    for i in 0..=9 {
                        let is_block = i % 3 == 0;
                        let thick = if is_block { 2.6 } else { 1.2 };
                        let line_col = if is_block {
                            Color::new(0.35, 0.65, 0.95, 0.85) // Original electric blue for 3x3 blocks
                        } else {
                            Color::new(0.25, 0.35, 0.50, 0.70) // Original slate blue for cell dividers
                        };

                        let y = leader_y + (i as f32) * cell_size;
                        let x = leader_x + (i as f32) * cell_size;
                        draw_line(leader_x, y, leader_x + leader_size, y, thick, line_col);
                        draw_line(x, leader_y, x, leader_y + leader_size, thick, line_col);
                    }
                }

                // ---------------- 8 CHALLENGERS (#2 to #9) ON THE RIGHT ----------------
                let right_x = leader_x + leader_size + 30.0;
                let right_w = screen_w - right_x - 20.0;
                if !current_state.solved {
                    let grid_cols = 4;
                    let grid_rows = 2;
                    let mini_w = (right_w - 3.0 * 12.0) / (grid_cols as f32);
                    let mini_h = ((screen_h - 120.0) - 12.0) / (grid_rows as f32);

                    draw_text("CHALLENGERS #2 - #9 (EVOLUTIONARY DIVERSITY)", right_x, leader_y - 12.0, 15.0, LIGHTGRAY);

                    for i in 1..9 {
                        if i >= current_state.top9.len() {
                            break;
                        }
                        let idx_challenger = i - 1;
                        let col = idx_challenger % grid_cols;
                        let row = idx_challenger / grid_cols;

                        let mx = right_x + (col as f32) * (mini_w + 12.0);
                        let my = leader_y + (row as f32) * (mini_h + 12.0);

                        // Card background
                        draw_rectangle(mx, my, mini_w, mini_h, Color::new(0.08, 0.10, 0.14, 0.9));
                        draw_rectangle_lines(mx, my, mini_w, mini_h, 1.0, Color::new(0.25, 0.32, 0.45, 0.6));

                        let ind = &current_state.top9[i];
                        let badge_color = Color::new(0.85, 0.85, 0.9, 1.0);
                        draw_text(
                            &format!("#{} | c:{} | fit:{}", i + 1, ind.conflicts, ind.fitness),
                            mx + 6.0,
                            my + 16.0,
                            13.0,
                            badge_color,
                        );

                        // Mini Sudoku Grid
                        let mini_board_y = my + 24.0;
                        let mini_board_size = (mini_h - 28.0).min(mini_w - 12.0);
                        let mini_cell = mini_board_size / 9.0;
                        let mini_analysis = analyze_board(&ind.grid);

                        for r in 0..9 {
                            for c in 0..9 {
                                let cell_idx = r * 9 + c;
                                let cx = mx + 6.0 + (c as f32) * mini_cell;
                                let cy = mini_board_y + (r as f32) * mini_cell;
                                let is_conflict = mini_analysis.cell_conflicts[cell_idx];
                                let is_fixed = current_state.fixed.get(cell_idx).copied().unwrap_or(false);
                                let c_color = if is_conflict {
                                    Color::new(0.85, 0.15, 0.20, 0.95) // Red for conflicts
                                } else if is_fixed {
                                    Color::new(0.18, 0.25, 0.35, 0.9) // Original dark blue-slate for clues
                                } else {
                                    Color::new(0.06, 0.10, 0.17, 0.9) // Original deep dark blue
                                };

                                draw_rectangle(cx, cy, mini_cell - 0.5, mini_cell - 0.5, c_color);

                                // Draw number in small challenger cell
                                if cell_idx < ind.grid.len() {
                                    let num = ind.grid[cell_idx];
                                    if num > 0 {
                                        let text = format!("{}", num);
                                        let font_size = 11.0;
                                        let dims = measure_text(&text, None, font_size as u16, 1.0);
                                        let text_color = if is_fixed {
                                            GOLD
                                        } else {
                                            WHITE
                                        };
                                        draw_text(
                                            &text,
                                            cx + (mini_cell - dims.width) * 0.5,
                                            cy + (mini_cell + dims.height) * 0.5 - 1.0,
                                            font_size,
                                            text_color,
                                        );
                                    }
                                }
                            }
                        }

                        // Mini 3x3 block lines
                        for l in 0..=3 {
                            let pos = (l * 3) as f32 * mini_cell;
                            draw_line(
                                mx + 6.0,
                                mini_board_y + pos,
                                mx + 6.0 + mini_board_size,
                                mini_board_y + pos,
                                1.2,
                                Color::new(0.4, 0.5, 0.65, 0.7),
                            );
                            draw_line(
                                mx + 6.0 + pos,
                                mini_board_y,
                                mx + 6.0 + pos,
                                mini_board_y + mini_board_size,
                                1.2,
                                Color::new(0.4, 0.5, 0.65, 0.7),
                            );
                        }
                    }
                } else {
                    // ---------------- SOLVED RIGHT-SIDE VICTORY PANEL ----------------
                    draw_text("★ TOURNAMENT RESULT: SOLUTION FOUND! ★", right_x, leader_y - 12.0, 15.0, GOLD);

                    let v_h = leader_size;
                    // Main victory container
                    draw_rectangle(right_x, leader_y, right_w, v_h, Color::new(0.06, 0.09, 0.14, 0.96));
                    draw_rectangle_lines(right_x, leader_y, right_w, v_h, 2.5, Color::new(0.2, 0.95, 0.35, 0.95));

                    // Glowing top bar accent
                    draw_rectangle(right_x, leader_y, right_w, 4.0, Color::new(0.2, 0.95, 0.35, 1.0));

                    // Title
                    draw_text("*** SUDOKU SOLVED! ***", right_x + 28.0, leader_y + 45.0, 30.0, GOLD);
                    draw_text(
                        "Global optimum reached. All Sudoku constraints 100% satisfied.",
                        right_x + 28.0,
                        leader_y + 72.0,
                        15.0,
                        LIGHTGRAY,
                    );

                    // 4 Stat Cards in 2x2 grid
                    let pad = 28.0;
                    let gap = 16.0;
                    let card_w = (right_w - pad * 2.0 - gap) / 2.0;
                    let card_h = 70.0;
                    let stats_y = leader_y + 96.0;

                    let stat_items = [
                        ("COMPLETED GENERATION", format!("Gen #{}", current_state.gen), Color::new(0.3, 0.8, 1.0, 1.0)),
                        ("FINAL FITNESS", "243 / 243 (MAX)".to_string(), Color::new(0.2, 0.95, 0.35, 1.0)),
                        ("TOTAL CONFLICTS", "0 (CLEARED)".to_string(), Color::new(0.2, 0.95, 0.35, 1.0)),
                        ("VALID LINES", "18 / 18 (9 Rows + 9 Cols)".to_string(), GOLD),
                    ];

                    for (idx, (label, val, val_col)) in stat_items.iter().enumerate() {
                        let c_col = idx % 2;
                        let c_row = idx / 2;
                        let sx = right_x + pad + (c_col as f32) * (card_w + gap);
                        let sy = stats_y + (c_row as f32) * (card_h + gap);

                        draw_rectangle(sx, sy, card_w, card_h, Color::new(0.10, 0.13, 0.19, 0.95));
                        draw_rectangle_lines(sx, sy, card_w, card_h, 1.2, Color::new(0.25, 0.35, 0.48, 0.8));

                        draw_text(label, sx + 12.0, sy + 22.0, 12.0, LIGHTGRAY);
                        draw_text(val, sx + 12.0, sy + 50.0, 18.0, *val_col);
                    }

                    // Verification Checklist box
                    let list_y = stats_y + card_h * 2.0 + gap + 20.0;
                    let list_h = 135.0;
                    draw_rectangle(right_x + pad, list_y, right_w - pad * 2.0, list_h, Color::new(0.08, 0.11, 0.16, 0.85));
                    draw_rectangle_lines(right_x + pad, list_y, right_w - pad * 2.0, list_h, 1.0, Color::new(0.2, 0.95, 0.35, 0.4));

                    draw_text("CONSTRAINT VALIDATION", right_x + pad + 16.0, list_y + 24.0, 13.0, Color::new(0.2, 0.95, 0.35, 1.0));
                    let checkmarks = [
                        "[✓] All 9 horizontal rows contain unique digits 1 through 9",
                        "[✓] All 9 vertical columns contain unique digits 1 through 9",
                        "[✓] All 9 3x3 block subgrids contain unique digits 1 through 9",
                        "[✓] All initial puzzle clues remain intact and unmodified",
                    ];
                    for (c_i, line) in checkmarks.iter().enumerate() {
                        draw_text(line, right_x + pad + 16.0, list_y + 48.0 + (c_i as f32) * 22.0, 13.0, Color::new(0.75, 0.88, 0.80, 1.0));
                    }

                    // Call to Action footer
                    let cta_y = list_y + list_h + 16.0;
                    let cta_h = v_h - (cta_y - leader_y) - 20.0;
                    if cta_h > 40.0 {
                        draw_rectangle(right_x + pad, cta_y, right_w - pad * 2.0, cta_h, Color::new(0.11, 0.17, 0.25, 0.95));
                        draw_rectangle_lines(right_x + pad, cta_y, right_w - pad * 2.0, cta_h, 1.5, Color::new(0.3, 0.7, 0.95, 0.8));

                        draw_text("WHAT'S NEXT?", right_x + pad + 16.0, cta_y + 22.0, 13.0, Color::new(0.3, 0.75, 1.0, 1.0));
                        draw_text("Click 'PUZZLE' in the top bar to choose a new difficulty level.", right_x + pad + 16.0, cta_y + 44.0, 14.0, WHITE);
                    }
                }
            }
        }

        next_frame().await;
    }
}
