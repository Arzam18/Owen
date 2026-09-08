#include "marrow.h"
#include <cmath>
#include <algorithm>
#include <limits>

namespace owen2::search {

double MarrowTree::ucb_score(const MarrowNode* parent, const MarrowNode* child, int parentVisits) const {
    double proven_bonus = (child->depth >= cfg_.classical_depth) ? cfg_.proven_bonus : 1.0;
    if(child->is_proven_win) return 1e12 * proven_bonus;
    if(child->is_proven_loss) return -1e12;
    if(child->visits==0){
        // Progressive bias: keep shallow-first but break ties by prior so good
        // moves like Nf3/Nc3 are tried before h4/f3 within the same depth.
        return 5e8 - child->depth*1e6 + child->prior * 1e5;
    }
    double q = child->q();
    double q_parent = -q;
    // depth-decayed exploration: slower decay past classical_depth for long-depth search
    double C_eff;
    if(child->depth < cfg_.classical_depth)
        C_eff = cfg_.C * std::pow(cfg_.C_decay, child->depth);
    else
        C_eff = cfg_.C * std::pow(cfg_.C_decay, cfg_.classical_depth) * std::pow(cfg_.long_depth_C_decay, child->depth - cfg_.classical_depth);
    double explore = C_eff * std::sqrt(std::log(double(parentVisits+1)) / double(child->visits));
    double prior_term = cfg_.policy_weight * child->prior * std::sqrt(double(parentVisits)) / (1+child->visits);
    // history heuristic — quiet moves that caused beta cuts get boost (scales to classical)
    double hist = cfg_.history_weight * std::tanh(child->history / 8192.0);
    // progressive widening penalty: beyond base, need more visits to be considered
    // at long depth, widen more (explore forks) but with proof bonus
    int widen = cfg_.prog_widen_base + (child->depth >= cfg_.classical_depth ? 2 : 0);
    double widen_pen = 0;
    if((int)parent->children.size() > widen && parent->visits < 800){
        // find child's rank
        int rank=0; for(auto &c: parent->children) if(c.get()==child) break; else rank++;
        if(rank >= widen) widen_pen = -0.35 * (rank - widen + 1) * (1.0 - double(parent->visits)/800.0);
    }
    // long-depth Q stabilizer: reduce noisy Q at deep plies
    double depth_stabilize = 1.0 + 0.02 * std::max(0, child->depth - cfg_.classical_depth);
    return q_parent / depth_stabilize + explore + prior_term + hist + widen_pen;
}

void MarrowTree::expand_node(MarrowNode* node, const Position& pos){
    if(node->expanded) return;
    auto moves = generate_legal(pos);
    if(moves.empty()){
        node->is_terminal=true;
        node->expanded=true;
        if(pos.in_check()) node->terminal_value = mated_in(pos.ply());
        else node->terminal_value = VALUE_DRAW;
        node->is_proven_loss = pos.in_check();
        return;
    }
    // TT move is only used as a tie-breaker/ordering hint, NOT a +10000
    // domination that collapses priors to delta after reuse. We cap the TT
    // bonus to a modest value so multipv-style visit rebalancing still works
    // and so a stale TT move cannot permanently pin one child at prior ~1.0
    // (that was the "root b1c3 -> g1f3 flip then stuck" bug).
    bool hit; TTEntry* e = tt_.probe(pos.key(), hit);
    Move ttMove = (hit? e->move : 0);

    struct Scored { Move m; int score; double hist; int pst; };
    // Tiny hand-crafted root bias so f3/f6 never leads when everything
    // else is equal (unloaded/untrained net or flat eval line). Keyed to
    // startpos geometry; deeper positions are dominated by eval/history.
    auto pst_move_bonus = [&](Move m)->int{
        Square from = move_from(m), to = move_to(m);
        // penalize f-pawn pushes to f3/f4/f6/f5 at the root; reward knights/centre pawns
        Color us = pos.side_to_move();
        if(type_of(pos.piece_on(from))==PAWN && file_of(from)==5){
            // f-file pawn move is terrible in the opening
            if((us==WHITE && rank_of(to)==2) || (us==BLACK && rank_of(to)==5)) return -95; // f3/f6
            if((us==WHITE && rank_of(to)==3) || (us==BLACK && rank_of(to)==4)) return -70; // f4/f5
        }
        if(type_of(pos.piece_on(from))==KNIGHT){
            int tf=file_of(to), tr=rank_of(to);
            if(us==WHITE && tf==5&&tr==2) return 52; // g1f3 — world-class fave
            if(us==WHITE && tf==2&&tr==2) return 36; // b1c3
            if(us==BLACK && tf==5&&tr==5) return 52; // g8f6
            if(us==BLACK && tf==2&&tr==5) return 36; // b8c6
        }
        if(type_of(pos.piece_on(from))==PAWN){
            // centre pawns e4/d4/c4
            int tf=file_of(to), tr=rank_of(to);
            if(us==WHITE && tr==3 && (tf==4||tf==3)) return 28; // e4,d4
            if(us==BLACK && tr==4 && (tf==4||tf==3)) return 28;
            if(us==WHITE && tr==2 && (tf==4||tf==3)) return 18; // e3/d3 second best
            if(us==BLACK && tr==5 && (tf==4||tf==3)) return 18;
        }
        // bishops to good squares (c4/f4 etc) slight bonus
        if(type_of(pos.piece_on(from))==BISHOP){
            int tf=file_of(to), tr=rank_of(to);
            if((us==WHITE && tr>=2 && tr<=3) || (us==BLACK && tr>=4 && tr<=5)){
                if(tf>=2 && tf<=5) return 10;
            }
        }
        return 0;
    };
    std::vector<Scored> scored; scored.reserve(moves.size());
    for(Move m: moves){
        int s=0;
        if(m==ttMove) s+=220; // modest TT bias, not 10000
        if(is_capture(m)) s+= 800 + (int)type_of(pos.piece_on(move_to(m)))*100;
        if(is_promo(m)) s+= 900;
        int f = move_from(m), t = move_to(m);
        double h = (f<64 && t<64) ? history_[f][t] : 0;
        if(!is_capture(m) && !is_promo(m)) s += int(std::tanh(h/4096.0)*120);
        int pst = pst_move_bonus(m);
        s += pst;
        scored.push_back({m,s,h,pst});
    }
    // Stable tie-break: equal scores keep legality order (g1f3 before f2f3 depends on movegen) but our pst
    // already separates f3; use stable_sort so deterministic.
    std::stable_sort(scored.begin(), scored.end(), [](auto& a, auto& b){return a.score>b.score;});
    double maxS = scored.empty()?0:scored[0].score;
    double sum=0; std::vector<double> ex(scored.size());
    for(size_t i=0;i<scored.size();++i){ ex[i]=std::exp((scored[i].score-maxS)/400.0); sum+=ex[i]; }
    // Clamp any single prior to 0.30 so one child can't dominate UCB exploration
    // forever after a lucky TT hit. This lets Marrow actually search.
    for(auto &v: ex) v = std::min(v / sum, 0.30);
    // Renormalize after clamp (rare, but keeps sum==1)
    { double s2=0; for(auto v: ex) s2+=v; if(s2>1e-9) for(auto &v: ex) v/=s2; sum=1.0; }
    node->children.reserve(scored.size());
    for(size_t i=0;i<scored.size();++i){
        auto child = std::make_unique<MarrowNode>();
        child->move = scored[i].m;
        child->prior = ex[i];
        child->history = scored[i].hist;
        child->depth = node->depth + 1;
        node->children.push_back(std::move(child));
    }
    node->expanded=true;
}

std::vector<MarrowTree::SelectFrame> MarrowTree::select_path(Position& pos, std::vector<MarrowNode*>& path){
    std::vector<SelectFrame> frames;
    MarrowNode* cur = root_.get();
    path.push_back(cur);
    while(cur->expanded && !cur->is_terminal && !cur->children.empty()){
        // check proven — if proven win exists, go there immediately (classical: forced mate)
        int provenIdx=-1;
        for(size_t i=0;i<cur->children.size();++i) if(cur->children[i]->is_proven_win){ provenIdx=(int)i; break; }
        int bestIdx = provenIdx;
        double bestScore = -1e100;
        if(bestIdx<0){
            for(size_t i=0;i<cur->children.size();++i){
                if(cur->children[i]->is_proven_loss) continue;
                double s = ucb_score(cur, cur->children[i].get(), cur->visits);
                if(s > bestScore){ bestScore=s; bestIdx=(int)i; }
            }
        }
        if(bestIdx<0){
            // all proven losses — pick least bad
            for(size_t i=0;i<cur->children.size();++i){
                double s = ucb_score(cur, cur->children[i].get(), cur->visits);
                if(s > bestScore){ bestScore=s; bestIdx=(int)i; }
            }
        }
        if(bestIdx<0) break;
        frames.push_back({cur, bestIdx});
        MarrowNode* nxt = cur->children[bestIdx].get();
        pos.do_move(nxt->move);
        path.push_back(nxt);
        cur = nxt;
        if(!cur->expanded) break;
        if(cur->visits==0) break;
        // LMR-style early stop: quiet deep nodes need more visits before deepening (scale to classical)
        if(cur->depth >= 8 && !is_capture(cur->move) && !is_promo(cur->move) && cur->visits < 3) break;
    }
    return frames;
}

Value MarrowTree::evaluate_leaf(const Position& pos){
    if(pos.is_draw()) return VALUE_DRAW;
    // Don't generate moves twice: evaluate() is the expensive part, movegen was already
    // done by the caller when checking terminal; pass in a flag if needed.
    // For now keep correct but avoid double movegen by early evaluate.
    int v = nnue::g_network.evaluate(pos);
    auto moves = generate_legal(pos);
    if(moves.empty()){
        if(pos.in_check()) return mated_in(pos.ply());
        return VALUE_DRAW;
    }
    return Value(v);
}

void MarrowTree::backup(std::vector<MarrowNode*>& path, Value leafValue){
    Value cur = leafValue;
    bool isWin = (leafValue > 9000) || (leafValue == VALUE_DRAW && false);
    // history update: if leaf was good for side to move, reward the move that led there
    for(int i=(int)path.size()-1; i>=0; --i){
        MarrowNode* n = path[i];
        n->visits++;
        n->total_value += double(cur);
        // proof propagation: mate scores propagate as proven
        if(std::abs(int(cur)) > 9000){
            if(cur > 0) n->is_proven_win = true;
            else if(cur < -9000) n->is_proven_loss = true;
        }
        // history: bonus to move that led to good leaf (classical heuristic)
        if(i>0 && cur > 200){
            Move m = path[i]->move;
            int f = move_from(m), t = move_to(m);
            if(f<64 && t<64 && !is_capture(m)){
                // depth-weighted history — deeper good moves get more
                int bonus = 512 * (path[i]->depth + 1);
                history_[f][t] += bonus;
                // age decay to keep bounded
                if(history_[f][t] > 16000) for(int a=0;a<64;++a) for(int b=0;b<64;++b) history_[a][b]/=2;
            }
        }
        cur = Value(-cur);
    }
}

Move MarrowTree::search_until(const std::function<bool()>& stop,
                              std::function<void(int,Value,Move)> on_info)
{
    expand_node(root_.get(), rootPos_);
    if(root_->children.empty()) return 0;
    if(root_->children.size()==1) return root_->children[0]->move;

    int lastInfoVisits=0;
    // Reuse a single Position object and an undo snapshot via make/unmake
    // copying the whole Position every iteration is wasteful; we use do_move/undo.
    while(!stop()){
        if(totalVisits_ >= cfg_.max_nodes) break;
        Position pos = rootPos_;
        std::vector<MarrowNode*> path; path.reserve(96);
        select_path(pos, path);
        MarrowNode* leaf = path.back();
        Value v;
        if(leaf->is_terminal){
            v = leaf->terminal_value;
        } else {
            if(!leaf->expanded){
                expand_node(leaf, pos);
                if(leaf->is_terminal) v = leaf->terminal_value;
                else v = evaluate_leaf(pos);
            } else {
                v = evaluate_leaf(pos);
            }
        }
        backup(path, v);
        totalVisits_++;

        // Throttle info callbacks: every 1024 visits (was 2048) and no extra stop() poll every 256
        if(on_info && totalVisits_ - lastInfoVisits >= 1024){
            lastInfoVisits = totalVisits_;
            MarrowNode* best=nullptr; int bestV=-1;
            for(auto &c: root_->children) if(c->visits > bestV){ bestV=c->visits; best=c.get(); }
            if(best) on_info(totalVisits_, Value(best->q()), best->move);
        }
    }
    MarrowNode* best=nullptr;
    for(auto &c: root_->children){
        if(!best || c->visits > best->visits || (c->visits==best->visits && c->q() > best->q()))
            best=c.get();
    }
    if(best) tt_.store(rootPos_.key(), Value(best->q()), 0, 0, best->move);
    return best? best->move : 0;
}

} // namespace owen2::search
