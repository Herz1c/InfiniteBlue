// Inside a world border of 1..8 x 1..8, the site's rules reduce to ordinary chess, so perft
// must equal the published chess numbers (https://www.chessprogramming.org/Perft_Results).
use icore::moves::perft;
use icore::pos::parse_icn_and_play;

/// Standard FEN -> ICN with an 8x8 border. Castling flags become "+" on king and rook;
/// pawns on their start rank get the double-step right.
fn fen_to_icn(fen: &str) -> String {
    let f: Vec<&str> = fen.split_whitespace().collect();
    let (board, turn, castling, ep) = (f[0], f[1], f[2], f[3]);
    let mut pieces = Vec::new();
    for (row, rank) in board.split('/').enumerate() {
        let y = 8 - row as i64;
        let mut x = 1;
        for c in rank.chars() {
            if let Some(d) = c.to_digit(10) {
                x += d as i64;
                continue;
            }
            let white = c.is_ascii_uppercase();
            let right = match c.to_ascii_lowercase() {
                'p' => y == if white { 2 } else { 7 },
                'k' => (white && castling.contains(['K', 'Q'])) || (!white && castling.contains(['k', 'q'])),
                'r' => {
                    let (home, short, long) = if white { (1, 'K', 'Q') } else { (8, 'k', 'q') };
                    y == home && ((x == 8 && castling.contains(short)) || (x == 1 && castling.contains(long)))
                }
                _ => false,
            };
            pieces.push(format!("{c}{x},{y}{}", if right { "+" } else { "" }));
            x += 1;
        }
    }
    let ep = if ep == "-" {
        String::new()
    } else {
        let b = ep.as_bytes();
        format!("{},{} ", (b[0] - b'a' + 1), (b[1] - b'0'))
    };
    format!("{turn} {ep}0/100 1 (8|1) 1,8,1,8 {}", pieces.join("|"))
}

fn check(fen: &str, expected: &[u64]) {
    let pos = parse_icn_and_play(&fen_to_icn(fen)).unwrap();
    for (i, &want) in expected.iter().enumerate() {
        let got = perft(&pos, i as u32 + 1);
        assert_eq!(got, want, "{fen} depth {}", i + 1);
    }
}

#[test]
fn start_position() {
    check("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", &[20, 400, 8902, 197_281, 4_865_609]);
}

#[test]
fn kiwipete() {
    check("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", &[48, 2039, 97_862, 4_085_603]);
}

#[test]
fn position_3() {
    check("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", &[14, 191, 2812, 43_238, 674_624]);
}

#[test]
fn position_4() {
    check("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", &[6, 264, 9467, 422_333]);
}

#[test]
fn position_5() {
    check("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", &[44, 1486, 62_379, 2_103_487]);
}

#[test]
fn position_6() {
    check("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", &[46, 2079, 89_890, 3_894_594]);
}
