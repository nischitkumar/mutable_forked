#pragma once

#include <algorithm>
#include <cstring>
#include <iostream>
#include <utility>
#include <vector>
#include <mutable/util/exception.hpp>
#include <mutable/util/macro.hpp>
#include <mutable/util/Visitor.hpp>


namespace m {

// Forward declarations
struct Table;
struct Attribute;
struct Schema;

namespace idx {

/** An enum class that lists all supported index methods. */
enum class IndexMethod { Array };

// Forward declarations
struct IndexVisitor;
struct ConstIndexVisitor;

/** The base class for indexes. */
struct IndexBase
{
    virtual ~IndexBase() { }

    /** Bulkloads the index by executing a query on \p table using \p key_schema. */
    virtual void bulkload(const Table &table, const Schema &key_schema) = 0;
    /* Returns the number of entries in the index. */
    virtual std::size_t num_entries() const = 0;
    /** Returns the `IndexMethod` of the index. */
    virtual IndexMethod method() const = 0;

    virtual void dump(std::ostream &out) const = 0;
    virtual void dump() const = 0;

    virtual void accept(IndexVisitor &v) = 0;
    virtual void accept(ConstIndexVisitor &v) const = 0;

    protected:
    /** Constructs a query string to select all attributes in \p schema from \p table. */
    static std::string build_query(const Table &table, const Schema &schema);
};

/** A simple index based on a sorted array that maps keys to their `tuple_id`. */
template<typename Key>
struct ArrayIndex : IndexBase
{
    using key_type = Key;
    using value_type = std::size_t;
    using entry_type = std::pair<key_type, value_type>;
    using container_type = std::vector<entry_type>;
    using const_iterator = typename container_type::const_iterator;

    private:
    container_type data_; ///< A vector holding the index entries consisting of pairs of key and value
    bool finalized_; ///< flag to signalize whether index is finalized, i.e. array is sorted

    /** Custom comparator class to handle the special case of \tparam key_type being `const char*`. */
    struct {
        bool operator()(const entry_type lhs, const entry_type rhs) {
            if constexpr(std::same_as<key_type, const char*>) return std::strcmp(lhs.first, rhs.first) < 0;
            else return lhs.first < rhs.first;
        }
    } cmp;

    public:
    ArrayIndex() : finalized_(false) { }

    /** Bulkloads the index from \p table on the key contained in \p key_schema by executing a query and adding one
     * entry after another.  The index is finalized in the end.  Throws `m::invalid_arguent` if \p key_schema contains
     * more than one entry or `key_type` and the attribute type of the entry in \p key_schema do not match. */
    void bulkload(const Table &table, const Schema &key_schema) override;

    /** Returns the `IndexMethod` of the index. */
    IndexMethod method() const override { return IndexMethod::Array; }

    /** Adds a single pair of \p key and \p value to the index.  Note that `finalize()` has to be called afterwards for
     * the vector to be sorted and the index to be usable. */
    void add(const key_type key, const value_type value);

    /** Sorts the underlying vector and flags the index as finalized. */
    void finalize() {
        std::sort(data_.begin(), data_.end(), cmp);
        finalized_ = true;
    }

    /** Returns an iterator pointing to the first entry of the vector such that `entry.key` < \p key is `false`, i.e.
     * that is greater than or equal to \p key, or `end()` if no such element is found.  Throws `m::exception` if the
     * index is not finalized. */
    const_iterator lower_bound(const key_type key) const {
        if (not finalized_) throw m::exception("Index is not finalized.");
        return std::lower_bound(data_.begin(), data_.end(), entry_type{key, value_type()}, cmp);
    }

    /** Returns an iterator pointing to the first entry of the vector such that `entry.key` < \p key is `true`, i.e.
     * that is strictly greater than \p key, or `end()` if no such element is found.  Throws `m::exception` if the index
     * is not finalized. */
    const_iterator upper_bound(const key_type key) const {
        if (not finalized_) throw m::exception("Index is not finalized.");
        return std::upper_bound(data_.begin(), data_.end(), entry_type{key, value_type()}, cmp);
    }

    /** Returns an iterator pointing to the first entry of the index. */
    const_iterator begin()  const { return data_.cbegin(); }
    const_iterator cbegin() const { return data_.cbegin(); }
    /** Returns an interator pointing to the first element following the last entry of the index. */
    const_iterator end() const  { return data_.cend(); }
    const_iterator cend() const { return data_.cend(); }

    /** Returns the number of entries in the index. */
    std::size_t num_entries() const override { return data_.size(); }

    /** Returns `true` iff the index is currently finalized. */
    bool finalized() const { return finalized_; }

    void dump(std::ostream &out) const override { out << "ArrayIndex<" << typeid(key_type).name() << '>' << std::endl; }
    void dump() const override { dump(std::cerr); }

    void accept(IndexVisitor &v) override;
    void accept(ConstIndexVisitor &v) const override;
};

#define M_INDEX_LIST_TEMPLATED(X) \
    X(m::idx::ArrayIndex<bool>) \
    X(m::idx::ArrayIndex<int8_t>) \
    X(m::idx::ArrayIndex<int16_t>) \
    X(m::idx::ArrayIndex<int32_t>) \
    X(m::idx::ArrayIndex<int64_t>) \
    X(m::idx::ArrayIndex<float>) \
    X(m::idx::ArrayIndex<double>) \
    X(m::idx::ArrayIndex<const char*>)

M_DECLARE_VISITOR(IndexVisitor, IndexBase, M_INDEX_LIST_TEMPLATED)
M_DECLARE_VISITOR(ConstIndexVisitor, const IndexBase, M_INDEX_LIST_TEMPLATED)

}

}

// explicit instantiation declarations
#define DECLARE(CLASS) \
    extern template struct CLASS;
M_INDEX_LIST_TEMPLATED(DECLARE)
#undef DECLARE
