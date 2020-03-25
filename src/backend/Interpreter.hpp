#pragma once

#include "backend/Backend.hpp"
#include "catalog/Schema.hpp"
#include "IR/Operator.hpp"
#include "IR/OperatorVisitor.hpp"
#include "IR/Tuple.hpp"
#include "util/macro.hpp"
#include <unordered_map>


namespace db {

/** A block of size `N` contains `N` tuples.  */
template<std::size_t N>
struct Block
{
    static constexpr std::size_t CAPACITY = N; ///< the capacity of a block

    private:
    template<bool C>
    struct the_iterator
    {
        static constexpr bool IsConst = C;

        using block_t = std::conditional_t<IsConst, const Block, Block>;
        using reference = std::conditional_t<IsConst, const Tuple&, Tuple&>;
        using pointer = std::conditional_t<IsConst, const Tuple*, Tuple*>;

        private:
        block_t &block_;
        uint64_t mask_;

        public:
        the_iterator(block_t &vec, uint64_t mask) : block_(vec), mask_(mask) { }

        bool operator==(the_iterator other) {
            insist(&this->block_ == &other.block_);
            return this->mask_ == other.mask_;
        }
        bool operator!=(the_iterator other) { return not operator==(other); }

        the_iterator & operator++() { mask_ = mask_ & (mask_ - 1UL); /* set lowest 1-bit to 0 */ return *this; }
        the_iterator operator++(int) { the_iterator clone(*this); operator++(); return clone; }

        std::size_t index() const { return __builtin_ctzl(mask_); }

        reference operator*() const { return block_[index()]; }
        pointer operator->() const { return &block_[index()]; }
    };

    public:
    using iterator = the_iterator<false>;
    using const_iterator = the_iterator<true>;

    private:
    std::array<Tuple, N> data_; ///< an array of the tuples of this `Block`; some slots may be unused
    uint64_t mask_ = 0x0; ///< a mast identifying which slots of `data_` are in use
    static_assert(N <= 64, "maximum block size exceeded");

    public:
    Block() = default;
    Block(const Block&) = delete;
    Block(Block&&) = delete;

    /** Create a new `Block` with tuples of `Schema` `schema`. */
    Block(Schema schema) {
        for (auto &t : data_)
            t = Tuple(schema);
    }

    /** Return a pointer to the underlying array of tuples. */
    Tuple * data() { return data_.data(); }
    /** Return a pointer to the underlying array of tuples. */
    const Tuple * data() const { return data_.data(); }

    /** Return the capacity of this `Block`. */
    static constexpr std::size_t capacity() { return CAPACITY; }
    /** Return the number of *alive* tuples in this `Block`. */
    std::size_t size() const { return __builtin_popcountl(mask_); }

    iterator begin() { return iterator(*this, mask_); }
    iterator end()   { return iterator(*this, 0UL); }
    const_iterator begin() const { return const_iterator(*this, mask_); }
    const_iterator end()   const { return const_iterator(*this, 0UL); }
    const_iterator cbegin() const { return const_iterator(*this, mask_); }
    const_iterator cend()   const { return const_iterator(*this, 0UL); }

    /** Returns an iterator to the tuple at index `index`. */
    iterator at(std::size_t index) {
        insist(index < capacity());
        return iterator(*this, mask_ & (-1UL << index));
    }
    /** Returns an iterator to the tuple at index `index`. */
    const_iterator at(std::size_t index) const { return const_cast<Block>(this)->at(index); }

    /** Check whether the tuple at the given `index` is alive. */
    bool alive(std::size_t index) const {
        insist(index < capacity());
        return mask_ & (1UL << index);
    }

    /** Returns `true` iff the block has no *alive* tuples, i.e.\ `size() == 0`. */
    bool empty() const { return size() == 0; }

    /** Returns the bit mask that identifies which tuples of this `Block` are alive. */
    uint64_t mask() const { return mask_; }
    /** Returns the bit mask that identifies which tuples of this `Block` are alive. */
    void mask(uint64_t new_mask) { mask_ = new_mask; }

    private:
    /** Returns a bit vector with left-most `capacity()` many bits set to `1` and the others set to `0`.  */
    static constexpr uint64_t AllOnes() { return -1UL >> (8 * sizeof(mask_) - capacity()); }

    public:
    /** Returns the tuple at index `index`.  The tuple must be *alive*!  */
    Tuple & operator[](std::size_t index) {
        insist(index < capacity(), "index out of bounds");
        insist(alive(index), "cannot access a dead tuple directly");
        return data_[index];
    }
    /** Returns the tuple at index `index`.  The tuple must be *alive*!  */
    const Tuple & operator[](std::size_t index) const { return const_cast<Block*>(this)->operator[](index); }

    /** Make all tuples in this `Block` *alive*. */
    void fill() { mask_ = AllOnes(); insist(size() == capacity()); }

    /** Erase the tuple at the given `index` from this `Block`. */
    void erase(std::size_t index) {
        insist(index < capacity(), "index out of bounds");
        setbit(&mask_, false, index);
    }
    /** Erase the tuple identified by `it` from this `Block`. */
    void erase(iterator it) { erase(it.index()); }
    /** Erase the tuple identified by `it` from this `Block`. */
    void erase(const_iterator it) { erase(it.index()); }

    /** Renders all tuples *dead* and removes their attributes.. */
    void clear() {
        mask_ = 0;
        for (auto &t : data_)
            t.clear();
    }

    /** Print a textual representation of this `Block` to `out`. */
    friend std::ostream & operator<<(std::ostream &out, const Block<N> &block) {
        out << "Block<" << block.capacity() << "> with " << block.size() << " elements:\n";
        for (std::size_t i = 0; i != block.capacity(); ++i) {
            out << "    " << i << ": ";
            if (block.alive(i))
                out << block[i];
            else
                out << "[dead]";
            out << '\n';
        }
        return out;
    }

    void dump(std::ostream &out) const
    {
        out << *this;
        out.flush();
    }
    void dump() const { dump(std::cerr); }
};

struct Interpreter;

/** Implements push-based evaluation of a pipeline in the plan. */
struct Pipeline : ConstOperatorVisitor
{
    friend struct Interpreter;

    private:
    Block<64> block_;

    public:
    Pipeline() { }

    Pipeline(const Schema &schema)
        : block_(schema)
    {
        block_.mask(1UL); // create one empty tuple in the block
    }

    Pipeline(Tuple &&t)
    {
        block_.mask(1UL);
        block_[0] = std::move(t);
    }

    void push(const Operator &pipeline_start) { (*this)(pipeline_start); }

    void clear() { block_.clear(); }

    using ConstOperatorVisitor::operator();
#define DECLARE(CLASS) void operator()(Const<CLASS> &op) override
    DECLARE(ScanOperator);
    DECLARE(CallbackOperator);
    DECLARE(PrintOperator);
    DECLARE(NoOpOperator);
    DECLARE(FilterOperator);
    DECLARE(JoinOperator);
    DECLARE(ProjectionOperator);
    DECLARE(LimitOperator);
    DECLARE(GroupingOperator);
    DECLARE(SortingOperator);
#undef DECLARE
};

/** Evaluates SQL operator trees on the database. */
struct Interpreter : Backend, ConstOperatorVisitor
{
    public:
    Interpreter() = default;

    void execute(const Operator &plan) const override { (*const_cast<Interpreter*>(this))(plan); }

    using ConstOperatorVisitor::operator();
#define DECLARE(CLASS) void operator()(Const<CLASS> &op) override
    DECLARE(ScanOperator);
    DECLARE(CallbackOperator);
    DECLARE(PrintOperator);
    DECLARE(NoOpOperator);
    DECLARE(FilterOperator);
    DECLARE(JoinOperator);
    DECLARE(ProjectionOperator);
    DECLARE(LimitOperator);
    DECLARE(GroupingOperator);
    DECLARE(SortingOperator);
#undef DECLARE

    static Value eval(const Constant &c)
    {
        errno = 0;
        switch (c.tok.type) {
            default: unreachable("illegal token");

            /* Null */
            case TK_Null:
                unreachable("NULL cannot be evaluated to a Value");

            /* Integer */
            case TK_OCT_INT:
                return int64_t(strtoll(c.tok.text, nullptr, 8));

            case TK_DEC_INT:
                return int64_t(strtoll(c.tok.text, nullptr, 10));

            case TK_HEX_INT:
                return int64_t(strtoll(c.tok.text, nullptr, 16));

            /* Float */
            case TK_DEC_FLOAT:
                return strtod(c.tok.text, nullptr);

            case TK_HEX_FLOAT:
                unreachable("not implemented");

            /* String */
            case TK_STRING_LITERAL: {
                std::string str(c.tok.text);
                auto substr = interpret(str);
                return Catalog::Get().pool(substr.c_str()); // return internalized string by reference
            }

            /* Boolean */
            case TK_True:
                return true;

            case TK_False:
                return false;
        }
        insist(errno == 0, "constant could not be parsed");
    }
};

}
