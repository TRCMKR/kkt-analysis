import os
import re
from collections import defaultdict

def extract_experiments_count(filename):
    """Извлекает количество экспериментов из имени файла."""
    match = re.search(r'_(\d+)exp', filename)
    return int(match.group(1)) if match else 1


def parse_result_file(filepath):
    """Парсит файл результатов, возвращает словарь ключ -> значение."""
    results = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or 'NO DATA' in line:
                continue
            if ':' in line:
                key, value = line.rsplit(':', 1)
                key = key.strip()
                value = float(value.strip())
                results[key] = value
    return results


def merge_results(folder_path):
    """Объединяет результаты из всех файлов в папке."""
    weighted_sums = defaultdict(float)
    total_weights = defaultdict(int)

    for filename in os.listdir(folder_path):
        if not filename.startswith('result_') or not filename.endswith('.txt'):
            continue

        filepath = os.path.join(folder_path, filename)
        experiments = extract_experiments_count(filename)
        results = parse_result_file(filepath)

        print(f"Обработка {filename}: {experiments} экспериментов, {len(results)} записей")

        for key, value in results.items():
            weighted_sums[key] += value * experiments
            total_weights[key] += experiments

    # Вычисляем взвешенное среднее
    merged = {}
    for key in weighted_sums:
        merged[key] = weighted_sums[key] / total_weights[key]

    return merged, total_weights


def save_merged_results(merged, weights, output_path):
    """Сохраняет объединённые результаты в файл."""

    # Сортируем ключи
    def sort_key(k):
        parts = k.split()
        return tuple(int(p) for p in parts)

    sorted_keys = sorted(merged.keys(), key=sort_key)

    with open(output_path, 'w', encoding='utf-8') as f:
        for key in sorted_keys:
            f.write(f"{key}: {merged[key]}\n")
            # f.write(f"{key}: {merged[key]:.10f} ({weights[key]} exp)\n")

    print(f"\nРезультаты сохранены в {output_path}")
    print(f"Всего уникальных комбинаций: {len(merged)}")


def main():
    folder_path = "квадрат укороченных кодов"
    output_path = os.path.join(folder_path, "merged_results.txt")

    merged, weights = merge_results(folder_path)
    save_merged_results(merged, weights, output_path)


if __name__ == "__main__":
    main()
