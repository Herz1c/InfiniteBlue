// Match referee: decides move legality and game end with the site's own rules code
// (ref/site, imported, never copied). JSON lines on stdin -> one JSON line per request on stdout.
//
//   {"cmd":"new","id":"g1","icn":"<start ICN, may include moves>"}  -> {"ok":true,"result":null|"1-0"|"0-1"|"1/2-1/2","condition":...}
//   {"cmd":"move","id":"g1","move":"x,y>x,y[=Q]"}                    -> {"ok":true,"legal":true,"result":...,"condition":...}
//                                                                     or {"ok":true,"legal":false,"reason":"..."}
//   {"cmd":"validate","icn":"<whole game ICN>"}                       -> same as "new" (all moves are checked)
//   {"cmd":"drop","id":"g1"}                                          -> {"ok":true}
//   {"cmd":"moves","icn":"..."}  -> the site's legal moves for the side to move (oracle for our movegen):
//       {"ok":true,"pieces":[{"from":[x,y],"individual":[[x,y],..],"sliding":{"dx,dy":[lo|null,hi|null]},"brute":bool}]}
//       Coordinates are decimal strings. A slide [lo,hi] counts steps along (dx,dy), negative = backwards;
//       null = unbounded. With "brute", each slide square still has to pass a check test (see "legal").
//   {"cmd":"legal","icn":"...","move":"x,y>x,y"} -> {"ok":true,"legal":bool}
// Any failure -> {"ok":false,"error":"..."}.

import * as readline from 'node:readline';

import icnconverter from '../../ref/site/src/shared/chess/logic/icn/icnconverter.js';
import movevalidation from '../../ref/site/src/shared/chess/logic/movevalidation.js';
import movepiece from '../../ref/site/src/shared/chess/logic/movepiece.js';
import wincondition from '../../ref/site/src/shared/chess/logic/wincondition.js';
import legalmoves from '../../ref/site/src/shared/chess/logic/legalmoves.js';
import boardutil from '../../ref/site/src/shared/chess/util/boardutil.js';
import typeutil, { rawTypes } from '../../ref/site/src/shared/chess/util/typeutil.js';
import gameformulator from '../../ref/site/src/client/scripts/esm/game/chess/gameformulator.js';

const games = new Map<string, any>();

function verdict(game: any): { result: string | null; condition: string | null } {
	const c = wincondition.getGameConclusion(game);
	if (c === undefined) return { result: null, condition: null };
	const result = c.victor === 1 ? '1-0' : c.victor === 2 ? '0-1' : '1/2-1/2';
	return { result, condition: c.condition };
}

function load(icn: string): any {
	const longFormat = icnconverter.ShortToLong_Format(icn);
	return gameformulator.formulateGame(longFormat, true); // throws on any illegal move
}

/** Every legal move of the side to move, as the site computes it (same loop as checkmate.ts). */
function allMoves(game: any): object[] {
	const { basegame, boardsim } = game;
	const out: object[] = [];
	for (const rType of Object.values(rawTypes)) {
		const range = boardsim.pieces.typeRanges.get(typeutil.buildType(rType as number, basegame.whosTurn));
		if (!range) continue;
		for (let idx = range.start; idx < range.end; idx++) {
			const piece = boardutil.getPieceFromIdx(boardsim.pieces, idx);
			if (!piece) continue;
			const moves: any = legalmoves.calculateAll(game, piece);
			const sliding: Record<string, (string | null)[]> = {};
			for (const [line, limits] of Object.entries(moves.sliding as Record<string, (bigint | null)[]>)) {
				sliding[line] = limits.map((v) => (v === null ? null : v.toString()));
			}
			out.push({
				from: piece.coords.map(String),
				individual: moves.individual.map((c: bigint[]) => [c[0]!.toString(), c[1]!.toString()]),
				sliding,
				brute: !!moves.brute,
			});
		}
	}
	return out;
}

function handle(req: any): object {
	if (req.cmd === 'moves') return { ok: true, pieces: allMoves(load(req.icn)) };
	if (req.cmd === 'legal') {
		const check = movevalidation.isTokenMoveLegal(load(req.icn), req.move);
		return { ok: true, legal: check.valid };
	}
	if (req.cmd === 'new' || req.cmd === 'validate') {
		const game = load(req.icn);
		if (req.cmd === 'new') games.set(req.id, game);
		return { ok: true, ...verdict(game) };
	}
	if (req.cmd === 'move') {
		const game = games.get(req.id);
		if (!game) return { ok: false, error: `unknown game ${req.id}` };
		const check = movevalidation.isTokenMoveLegal(game, req.move);
		if (!check.valid) return { ok: true, legal: false, reason: check.reason };
		movepiece.generateAndMakeMove(game, { ...check.tagged, token: req.move });
		return { ok: true, legal: true, ...verdict(game) };
	}
	if (req.cmd === 'drop') {
		games.delete(req.id);
		return { ok: true };
	}
	return { ok: false, error: `unknown cmd ${req.cmd}` };
}

const rl = readline.createInterface({ input: process.stdin });
rl.on('line', (line) => {
	if (!line.trim()) return;
	let out: object;
	try {
		out = handle(JSON.parse(line));
	} catch (e) {
		out = { ok: false, error: e instanceof Error ? e.message : String(e) };
	}
	process.stdout.write(JSON.stringify(out) + '\n');
});
