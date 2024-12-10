// Copyright 2019 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OPEN_SPIEL_GAMES_DARK_CHESS_H_
#define OPEN_SPIEL_GAMES_DARK_CHESS_H_

#include <array>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/games/chess/chess.h"
#include "open_spiel/games/chess/chess_board.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// Dark chess - imperfect information version of chess:
// https://en.wikipedia.org/wiki/Dark_chess
//
// Parameters:
//   "board_size"  int     Number of squares in each row and column (default: 8)
//   "fen"         string  String describing the chess board position in
//                         Forsyth-Edwards Notation. The FEN has to match
//                         the board size. Default values are available for
//                         board sizes 4 and 8.

namespace open_spiel {
namespace dark_chess {

// Constants.
inline constexpr int NumPlayers() { return 2; }
inline constexpr double LossUtility() { return -1; }
inline constexpr double DrawUtility() { return 0; }
inline constexpr double WinUtility() { return 1; }

// See action encoding below.
inline constexpr int NumDistinctActions() { return 4672; }

// https://math.stackexchange.com/questions/194008/how-many-turns-can-a-chess-game-take-at-maximum
inline constexpr int MaxGameLength() { return 17695; }

class DarkChessGame;
class DarkChessObserver;

// State of an in-play game.
class DarkChessState : public State {
 public:
  // Constructs a chess state at the given position in Forsyth-Edwards Notation.
  // https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation
  DarkChessState(std::shared_ptr<const Game> game, int board_size,
                 const std::string& fen);
  DarkChessState(const DarkChessState&) = default;

  DarkChessState& operator=(const DarkChessState&) = default;

  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : ColorToPlayer(Board().ToPlay());
  }
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action action) const override;
  std::string ToString() const override;

  bool IsTerminal() const override {
    return static_cast<bool>(MaybeFinalReturns());
  }

  std::vector<double> Returns() const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;

  void StateTensor(absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action action) override;

  // Current board.
  chess::ChessBoard& Board() { return current_board_; }
  const chess::ChessBoard& Board() const { return current_board_; }
  int BoardSize() const { return current_board_.BoardSize(); }

  // Starting board.
  chess::ChessBoard& StartBoard() { return start_board_; }
  const chess::ChessBoard& StartBoard() const { return start_board_; }

  std::vector<chess::Move>& MovesHistory() { return moves_history_; }
  const std::vector<chess::Move>& MovesHistory() const {
    return moves_history_;
  }

  std::vector<int> LastSeenPiece(chess::Color color, chess::PieceType piece_type) const {
    return last_seen_[chess::ToInt(color)][(int) piece_type - 1];
  }

  int TotalPieces() const { return total_pieces_; }

  void AddPiece() { total_pieces_++; }

  int Pawns(chess::Color color) const { return color == chess::Color::kWhite ? w_pawns : b_pawns; }
  int Rooks(chess::Color color) const { return color == chess::Color::kWhite ? w_rooks : b_rooks; }
  int Knights(chess::Color color) const { return color == chess::Color::kWhite ? w_knights : b_knights; }
  int Bishops(chess::Color color) const { return color == chess::Color::kWhite ? w_bishops : b_bishops; }
  int Queens(chess::Color color) const { return color == chess::Color::kWhite ? w_queens : b_queens; }

  void AddRook(chess::Color color) { color == chess::Color::kWhite ? w_rooks++ : b_rooks++; }
  void AddPawn(chess::Color color) { color == chess::Color::kWhite ? w_pawns++ : b_pawns++; }
  void AddKnight(chess::Color color) { color == chess::Color::kWhite ? w_knights++ : b_knights++; }
  void AddBishop(chess::Color color) { color == chess::Color::kWhite ? w_bishops++ : b_bishops++; }
  void AddQueen(chess::Color color) { color == chess::Color::kWhite ? w_queens++ : b_queens++; }

 protected:
  void DoApplyAction(Action action) override;

 private:
  friend class DarkChessObserver;

  // Draw can be claimed under the FIDE 3-fold repetition rule (the current
  // board position has already appeared twice in the history).
  bool IsRepetitionDraw() const;

  // Calculates legal actions and caches them. This is separate from
  // LegalActions() as there are a number of other methods that need the value
  // of LegalActions. This is a separate method as it's called from
  // IsTerminal(), which is also called by LegalActions().
  void MaybeGenerateLegalActions() const;

  absl::optional<std::vector<double>> MaybeFinalReturns() const;

  // We have to store every move made to check for repetitions and to implement
  // undo. We store the current board position as an optimization.
  std::vector<chess::Move> moves_history_;
  // We store the start board for history to support games not starting
  // from the start position.
  chess::ChessBoard start_board_;
  // We store the current board position as an optimization.
  chess::ChessBoard current_board_;

  // TODO(kubicon) This is specific for 8x8 chess and is here only for now.
  // For each color, each piece it stores the time when it was saw the last time at given position by opponent
  std::vector<std::vector<std::vector<int>>> last_seen_;

  int total_pieces_ = 0;
  int w_pawns = 0;
  int w_rooks = 0;
  int w_knights = 0;
  int w_bishops = 0;
  int w_queens = 0;
  int b_pawns = 0;
  int b_rooks = 0;
  int b_knights = 0;
  int b_bishops = 0;
  int b_queens = 0;
  // RepetitionTable records how many times the given hash exists in the history
  // stack (including the current board).
  // We are already indexing by board hash, so there is no need to hash that
  // hash again, so we use a custom passthrough hasher.
  class PassthroughHash {
   public:
    std::size_t operator()(uint64_t x) const {
      return static_cast<std::size_t>(x);
    }
  };
  using RepetitionTable = absl::flat_hash_map<uint64_t, int, PassthroughHash>;
  RepetitionTable repetitions_;
  mutable absl::optional<std::vector<Action>> cached_legal_actions_;
};

// Game object.
class DarkChessGame : public Game {
 public:
  explicit DarkChessGame(const GameParameters& params);
  int NumDistinctActions() const override {
    return chess::NumDistinctActions();
  }
  std::unique_ptr<State> NewInitialState() const override {
    // set total_pieces_ to the number of pieces on the board
    auto state = absl::make_unique<DarkChessState>(shared_from_this(), board_size_,
                                             fen_);
    for (int y = 0; y < board_size_; ++y) {
      for (int x = 0; x < board_size_; ++x) {
        chess::Square sq{x, y};
        chess::Piece piece = state->Board().at(sq);
        if (piece.color != chess::Color::kEmpty) {
          state->AddPiece();
          switch (piece.type) {
            case chess::PieceType::kPawn:
              state->AddPawn(piece.color);
              break;
            case chess::PieceType::kRook:
              state->AddRook(piece.color);
              break;
            case chess::PieceType::kKnight:
              state->AddKnight(piece.color);
              break;
            case chess::PieceType::kBishop:
              state->AddBishop(piece.color);
              break;
            case chess::PieceType::kQueen:
              state->AddQueen(piece.color);
              break;
            case chess::PieceType::kKing:
              break;
            default:
              SpielFatalError("Unknown piece type");
          }
        }
      }
    }
    return state;
  }
  int NumPlayers() const override { return chess::NumPlayers(); }
  double MinUtility() const override { return LossUtility(); }
  absl::optional<double> UtilitySum() const override { return DrawUtility(); }
  double MaxUtility() const override { return WinUtility(); }
  std::vector<int> ObservationTensorShape() const override {

    return {
      board_size_, board_size_,
      6 + // Player piece types
      6 + // Opponent piece types
      1 + // Empty tile
      1 + // Unknown tile
      1 + // information about the opponent's remaining pieces
      1 // 3 public: repetitions count in one-hot encoding, 2 public: side to play, 1 public: irreversible move counter -- a fraction of $n over 100, 4 private: left/right castling rights, one-hot encoded. In the dark chess original this require only 10 tiles, so we will have some overhead
    };

    // std::vector<int> shape{
    //     (13 +  // public boards:  piece types * colours + empty
    //      14)   // private boards: piece types * colours + empty + unknown
    //         * board_size_ * board_size_ +
    //     3 +    // public: repetitions count, one-hot encoding
    //     2 +    // public: side to play
    //     1 +    // public: irreversible move counter -- a fraction of $n over 100
    //     2 * 2  // private: left/right castling rights, one-hot encoded.
    // };
    // return shape;
  }

  std::vector<int> StateTensorShape() const override { 
    return {
      board_size_, board_size_,
      6 + // Player piece types
      6 + // Opponent piece types
      1 + // Empty tile
      // 1 + // Unknown tile not necessary in state tensor
      1 // 3 public: repetitions count in one-hot encoding, 2 public: side to play, 1 public: irreversible move counter -- a fraction of $n over 100, 4 private: left/right castling rights, one-hot encoded. In the dark chess original this require only 10 tiles, so we will have some overhead
      };
      // (13 /* piece types * colours + empty */ + 1 /* repetition count */ +
      //     1 /* side to move */ + 1 /* irreversible move counter */ +
      //     4 /* castling rights */) * chess::kMaxBoardSize * chess::kMaxBoardSize}; // Taken from chess.h
  }

  // Taken from chess.h
  // std::vector<int> StateTensorShape() const override {
  //   return {
  //     (13 /* piece types * colours + empty */ + 1 /* repetition count */ +
  //         1 /* side to move */ + 1 /* irreversible move counter */ +
  //         4 /* castling rights */) * chess::kMaxBoardSize * chess::kMaxBoardSize}; // Taken from chess.h
  // }

  int MaxGameLength() const override { return chess::MaxGameLength(); }
  std::shared_ptr<Observer> MakeObserver(
      absl::optional<IIGObservationType> iig_obs_type,
      const GameParameters& params) const;

  std::shared_ptr<DarkChessObserver> default_observer_;

 private:
  const int board_size_;
  const std::string fen_;
};

}  // namespace dark_chess
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_DARK_CHESS_H_
