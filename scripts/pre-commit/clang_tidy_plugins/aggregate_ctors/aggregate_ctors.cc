/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <clang-tidy/ClangTidy.h>
#include <clang-tidy/ClangTidyCheck.h>
#include <clang-tidy/ClangTidyModule.h>
#include <clang-tidy/ClangTidyModuleRegistry.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/Lex/Lexer.h>

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace clang::tidy::legate {

namespace {

// Names aren't important, what's important is that they are distinct
constexpr std::string_view CONSTRUCT_BIND{"construct"};
constexpr std::string_view CTOR_MEM_INIT_BIND{"ctor_mem_init"};

using namespace ast_matchers;  // NOLINT(google-build-using-namespace)

class AggregateConstructorCheck final : public ClangTidyCheck {
 public:
  using ClangTidyCheck::ClangTidyCheck;

  void registerMatchers(MatchFinder* finder) override;
  void check(const MatchFinder::MatchResult& result) override;

 private:
  void check_ctor_(const MatchFinder::MatchResult& result, const SourceRange& paren_range);
};

void AggregateConstructorCheck::registerMatchers(MatchFinder* finder)
{
  // Given
  //
  // auto x = int(1);
  // auto y = Foo(1, 2);
  // Bar z(1, 2, 3);
  // auto baz = Baz{1};
  // Bop bop{};
  //
  // Matches "int(1)", "Foo(1, 2)", and "z(1, 2, 3)" but not "Baz{1}" or "bop{}".
  finder->addMatcher(
    traverse(
      // Ignore compiler-generated nodes (e.g. implicit constructors or conversions).
      TK_IgnoreUnlessSpelledInSource,
      cxxConstructExpr(
        // Only match nodes that are written in the .cc (or main header files,
        // as determined by HeaderFilterRegex in .clang-tidy).
        isExpansionInMainFile(),
        // Ignore any ctors that are using {} already.
        unless(isListInitialization()),
        // Ignore implicit construction expressions that result from static, const, or
        // reinterpret-casting. For example:
        //
        // Foo{static_cast<Foo>(bar)};
        //
        // Has 2 construct exprs. The outer (`Foo{...}`) is obvious, but static_cast<Foo>(bar)
        // actually creates an implicit construction node to construct a temporary `Foo` as the
        // target of the cast.
        unless(hasParent(castExpr())),
        // And ignore specifically the 2 (size + value) and 1 (size)-argument std::vector
        // constructors. It's possible the author could have used {} (if the first argument is
        // not convertible to std::size_t), but sussing that out is a pretty error-prone
        // procedure.
        unless(allOf(
          // Is std::vector...
          hasDeclaration(cxxConstructorDecl(ofClass(hasName("std::vector")))),
          // ...and has either 1 or 2 arguments
          anyOf(argumentCountIs(1), argumentCountIs(2)))))
        .bind(CONSTRUCT_BIND)),
    this);

  // Given
  //
  // class Foo {
  //   Foo(int x, double y) : x_(x), y_{y}, some_class_(x, y) {}
  //
  //   int x_;
  //   double y_;
  //   SomeType some_class_;
  // };
  //
  // Matches "x_(x)" but not "y_{y}" or "some_class_(x, y)" (the latter would be matched by the
  // matcher above). In fact, the matcher above would NOT match "x_(x)" because the clang AST
  // actually considers this to be an "ImplicitCastExpr" and not a construction expressions due
  // to the fact that `int` is a basic type.
  //
  // This is why we need this second matcher.
  finder->addMatcher(
    traverse(
      // Ignore compiler-generated nodes (e.g. implicit constructors or conversions).
      TK_IgnoreUnlessSpelledInSource,
      // The nesting here is a little wonky, because cxxCtorInitializer() doesn't accept the
      // isExpandsionInMainFile() directly, so we need to attach it to the actual field
      // declaration ("int x_;" in the above code) for it to work.
      //
      // But note we bind to the cxxCtorInitializer() expression, so that's what we'll
      // ultimately get when we match.
      cxxCtorInitializer(forField(fieldDecl(isExpansionInMainFile())),
                         withInitializer(unless(anyOf(
                           // Ignore real construction expressions, they are matched above.
                           cxxConstructExpr(),
                           // And ignore any that are already using {} initializers.
                           initListExpr()))))
        .bind(CTOR_MEM_INIT_BIND)),
    this);
}

[[nodiscard]] std::string_view source_range_to_string(const SourceRange& range,
                                                      const ASTContext& ctx)
{
  auto&& sm = ctx.getSourceManager();
  // NOTE: sm.getSpellingLoc() used in case the range corresponds to a macro/preprocessed source.
  auto start_loc      = sm.getSpellingLoc(range.getBegin());
  auto last_token_loc = sm.getSpellingLoc(range.getEnd());
  auto end_loc        = Lexer::getLocForEndOfToken(last_token_loc, 0, sm, ctx.getLangOpts());

  return Lexer::getSourceText(
    clang::CharSourceRange::getCharRange(start_loc, end_loc), sm, ctx.getLangOpts());
}

void AggregateConstructorCheck::check_ctor_(const MatchFinder::MatchResult& result,
                                            const SourceRange& paren_range)
{
  auto s = source_range_to_string(paren_range, *result.Context);
  // Given
  //
  // SomeType(initializer, values, ...);
  //         ^^^^^^^^^^^^^^^^^^^^^^^^^^
  //               `paren_range`
  //
  // Then `s` is "(initializer, values, ...)".
  if (s.front() != '(') {
    const auto str = [&] {
      std::stringstream ss;

      ss
        << paren_range.printToString(result.Context->getSourceManager()) << ": Unexpected source \""
        << s << R"(", expected first character to be "(", found ")" << s.front()
        << "\" instead. There is a bug in how the matchers are defined in the Legate check, likely "
           "this should not have matched in the first place.";

      return std::move(ss).str();
    }();

    throw std::runtime_error{str};  // legate-lint: no-traced-throw
  }

  // Drop the leading '('...
  s.remove_prefix(1);
  // ...and the trailing ')'
  s.remove_suffix(1);

  diag(paren_range.getBegin(), "Constructor should use {} instead")
    << FixItHint::CreateReplacement(paren_range, std::string{"{"}.append(s).append("}"));
}

void AggregateConstructorCheck::check(const MatchFinder::MatchResult& result)
{
  auto&& nodes = result.Nodes;

  if (const auto* const construct = nodes.getNodeAs<CXXConstructExpr>(CONSTRUCT_BIND)) {
    // Argument-less default construction:
    //
    // SomeType foo;
    //
    // Will result in an invalid range when requesting the brace ranges (because they don't
    // exist). We could try and detect this using the matcher interface, but it seems quite
    // complicated to do.
    if (auto&& paren_loc = construct->getParenOrBraceRange(); paren_loc.isValid()) {
      try {
        check_ctor_(result, paren_loc);
      } catch (...) {
        construct->dump();
        throw;
      }
    }
    return;
  }

  if (const auto* const ctor_mem_init = nodes.getNodeAs<CXXCtorInitializer>(CTOR_MEM_INIT_BIND)) {
    const auto paren_loc =
      SourceRange{ctor_mem_init->getLParenLoc(), ctor_mem_init->getRParenLoc()};

    try {
      check_ctor_(result, paren_loc);
    } catch (...) {
      ctor_mem_init->getInit()->dump();
      throw;
    }

    return;
  }
}

// ==========================================================================================

class AggregateConstructorCheckModule final : public ClangTidyModule {
 public:
  static constexpr std::string_view CHECK_NAME{"legate-use-aggregate-constructor"};

  void addCheckFactories(ClangTidyCheckFactories& check_factories) override
  {
    check_factories.registerCheck<AggregateConstructorCheck>(CHECK_NAME);
  }
};

// Register the module using this statically initialized variable.
// NOLINTNEXTLINE(cert-err58-cpp)
ClangTidyModuleRegistry::Add<AggregateConstructorCheckModule> _{
  AggregateConstructorCheckModule::CHECK_NAME, "Checks for use of {} constructors"};

}  // namespace

}  // namespace clang::tidy::legate
