import pandas as pd
import numpy as np

from multi_column_sorter import (
    MultiColumnSorter,
    SortSpec,
    SortOrder,
    SortColumn,
    multi_sort,
)


def main():
    df = pd.DataFrame({
        "department": ["Engineering", "Engineering", "Sales", "Sales", "HR", "HR", "Engineering", "Sales"],
        "salary": [12000, 15000, 9000, 11000, 8000, 8500, 13000, 9500],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hank"],
        "join_date": pd.to_datetime([
            "2023-01-15", "2022-06-01", "2023-03-20",
            "2021-11-10", "2022-08-05", "2023-02-14",
            "2021-09-30", "2022-12-01",
        ]),
    })

    print("=" * 60)
    print("原始数据:")
    print("=" * 60)
    print(df.to_string(index=False))

    print("\n")
    print("=" * 60)
    print("示例1: 按 department 升序 + salary 降序")
    print("=" * 60)
    spec1 = SortSpec.from_list([
        ("department", SortOrder.ASC),
        ("salary", SortOrder.DESC),
    ])
    result1 = MultiColumnSorter(df).sort(spec1)
    print(result1.to_string(index=False))

    print("\n")
    print("=" * 60)
    print("示例2: 使用快捷函数 multi_sort — department 升序 + salary 降序")
    print("=" * 60)
    result2 = multi_sort(df, by=["department", "salary"], order=[SortOrder.ASC, SortOrder.DESC])
    print(result2.to_string(index=False))

    print("\n")
    print("=" * 60)
    print("示例3: 链式构建 — salary 降序 + join_date 升序")
    print("=" * 60)
    spec3 = SortSpec().add("salary", SortOrder.DESC).add("join_date", SortOrder.ASC)
    result3 = MultiColumnSorter(df).sort(spec3)
    print(result3.to_string(index=False))

    print("\n")
    print("=" * 60)
    print("示例4: 含空值的数据排序")
    print("=" * 60)
    df_na = pd.DataFrame({
        "group": ["A", "A", "B", "B", "A", "B"],
        "value": [10, np.nan, 30, np.nan, 20, 40],
    })
    print("含空值原始数据:")
    print(df_na.to_string(index=False))

    spec_na = SortSpec.from_list([
        ("group", SortOrder.ASC),
        ("value", SortOrder.DESC),
    ])
    result_na = MultiColumnSorter(df_na).sort(spec_na)
    print("\ngroup 升序 + value 降序 (NaN 在末尾):")
    print(result_na.to_string(index=False))

    spec_na_first = SortSpec.from_list([
        ("group", SortOrder.ASC, "first"),
        ("value", SortOrder.DESC, "first"),
    ])
    result_na_first = MultiColumnSorter(df_na).sort(spec_na_first)
    print("\ngroup 升序 + value 降序 (NaN 在开头):")
    print(result_na_first.to_string(index=False))

    print("\n")
    print("=" * 60)
    print("示例5: 验证列不存在时抛出异常")
    print("=" * 60)
    try:
        bad_spec = SortSpec.from_list([("nonexistent", SortOrder.ASC)])
        MultiColumnSorter(df).sort(bad_spec)
    except KeyError as e:
        print(f"捕获到预期异常: {e}")

    print("\n")
    print("=" * 60)
    print("示例6: 验证 by/order 长度不匹配时抛出异常")
    print("=" * 60)
    try:
        multi_sort(df, by=["department", "salary"], order=[SortOrder.ASC])
    except ValueError as e:
        print(f"捕获到预期异常: {e}")

    print("\n")
    print("=" * 60)
    print("示例7: 数字 vs 字符串混排 (自动类型安全处理)")
    print("=" * 60)
    df_mixed = pd.DataFrame({
        "category": ["A", "A", "B", "B", "A", "B"],
        "code": [100, "50", 200, "300", 150, "abc"],
    })
    print("混排原始数据:")
    print(df_mixed.to_string(index=False))

    spec_mixed = SortSpec.from_list([
        ("category", SortOrder.ASC),
        ("code", SortOrder.ASC),
    ])
    result_mixed = MultiColumnSorter(df_mixed).sort(spec_mixed)
    print("\ncategory 升序 + code 升序 (数字优先, 自动类型转换):")
    print(result_mixed.to_string(index=False))

    spec_mixed_desc = SortSpec.from_list([
        ("code", SortOrder.DESC),
    ])
    result_mixed_desc = MultiColumnSorter(df_mixed).sort(spec_mixed_desc)
    print("\ncode 降序:")
    print(result_mixed_desc.to_string(index=False))

    print("\n")
    print("=" * 60)
    print("示例8: 自定义排序 — 按\"高/中/低\"优先级排序")
    print("=" * 60)
    df_priority = pd.DataFrame({
        "task": ["修复登录Bug", "优化首页加载", "重构数据库", "更新README", "设计新API", "编写单元测试"],
        "priority": ["高", "中", "低", "中", "高", "低"],
        "assignee": ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"],
    })
    print("原始数据:")
    print(df_priority.to_string(index=False))

    spec_custom = SortSpec.from_list([
        ("priority", SortOrder.ASC, "last", ["高", "中", "低"]),
    ])
    result_custom = MultiColumnSorter(df_priority).sort(spec_custom)
    print('\n按 priority 自定义排序 (高→中→低):')
    print(result_custom.to_string(index=False))

    print("\n")
    print("=" * 60)
    print("示例9: 自定义排序 + 普通排序组合")
    print("=" * 60)
    spec_combo = SortSpec.from_list([
        ("priority", SortOrder.ASC, "last", ["高", "中", "低"]),
        ("assignee", SortOrder.ASC),
    ])
    result_combo = MultiColumnSorter(df_priority).sort(spec_combo)
    print("priority 自定义排序(高→中→低) + assignee 字母序:")
    print(result_combo.to_string(index=False))

    print("\n")
    print("=" * 60)
    print("示例10: 链式自定义排序 + 降序反转")
    print("=" * 60)
    spec_chain_custom = SortSpec() \
        .add("priority", SortOrder.DESC, "last", ["低", "中", "高"]) \
        .add("assignee", SortOrder.ASC)
    result_chain = MultiColumnSorter(df_priority).sort(spec_chain_custom)
    print("priority 降序(高→中→低) + assignee 字母序:")
    print(result_chain.to_string(index=False))

    print("\n")
    print("=" * 60)
    print("示例11: multi_sort 快捷函数 + custom_orders")
    print("=" * 60)
    result_multi_custom = multi_sort(
        df_priority,
        by=["priority", "assignee"],
        order=[SortOrder.ASC, SortOrder.ASC],
        custom_orders={"priority": ["高", "中", "低"]},
    )
    print("priority 自定义排序(高→中→低) + assignee 字母序:")
    print(result_multi_custom.to_string(index=False))

    print("\n")
    print("=" * 60)
    print("示例12: 自定义排序含未列出值和 NaN")
    print("=" * 60)
    df_unlisted = pd.DataFrame({
        "level": ["S", "A", "B", "C", "D", "SS", np.nan, "A"],
        "score": [95, 80, 70, 60, 50, 99, 0, 85],
    })
    print("原始数据:")
    print(df_unlisted.to_string(index=False))

    spec_unlisted = SortSpec.from_list([
        ("level", SortOrder.ASC, "last", ["SS", "S", "A", "B", "C", "D"]),
    ])
    result_unlisted = MultiColumnSorter(df_unlisted).sort(spec_unlisted)
    print("\nlevel 自定义排序(SS→S→A→B→C→D, 未列出值在后, NaN 在末尾):")
    print(result_unlisted.to_string(index=False))

    spec_unlisted_first = SortSpec.from_list([
        ("level", SortOrder.ASC, "first", ["SS", "S", "A", "B", "C", "D"]),
    ])
    result_unlisted_first = MultiColumnSorter(df_unlisted).sort(spec_unlisted_first)
    print("\nlevel 自定义排序(NaN 在开头, 未列出值在自定义值之后):")
    print(result_unlisted_first.to_string(index=False))


if __name__ == "__main__":
    main()
