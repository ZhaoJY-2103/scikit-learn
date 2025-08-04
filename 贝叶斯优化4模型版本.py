import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from lightgbm import LGBMRegressor  # 新增LightGBM
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')


# ==================== 1. 数据加载与预处理 ====================
def load_and_preprocess_data(file_path, target_column):
    """加载数据并进行预处理"""
    print(f"正在加载数据: {file_path}")
    data = pd.read_excel(file_path)

    if target_column not in data.columns:
        raise ValueError(f"目标列 '{target_column}' 不存在于数据中。可用的列: {list(data.columns)}")

    print(f"数据维度: {data.shape[0]} 行, {data.shape[1]} 列")
    print("前5行数据:")
    print(data.head())

    if data.isnull().sum().any():
        print("\n发现缺失值:")
        print(data.isnull().sum())
        data.fillna(method='ffill', inplace=True)
        print("已使用前向填充处理缺失值")

    X = data.drop(columns=[target_column])
    y = data[target_column]

    if 'timestamp' in data.columns:
        print("\n检测到时间戳列，将按时间排序...")
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data = data.sort_values('timestamp')
        X = data.drop(columns=[target_column, 'timestamp'])
        y = data[target_column]

    return X, y


# ==================== 2. 数据准备与泄露防护 ====================
def split_data_with_leakage_prevention(X, y, test_size=0.2):
    """时序数据隔离拆分（防止数据泄露）"""
    split_idx = int(len(X) * (1 - test_size))
    X_train_full = X.iloc[:split_idx]
    y_train_full = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    print(f"\n数据拆分详情:")
    print(f"- 总样本数: {len(X)}")
    print(f"- 训练集: 前{100 * (1 - test_size)}% ({len(X_train_full)}个样本)")
    print(f"- 测试集: 后{100 * test_size}% ({len(X_test)}个样本, 完全隔离)")

    return X_train_full, y_train_full, X_test, y_test


# ==================== 3. 贝叶斯优化器配置 ====================
def create_bayes_optimizer(model_class, param_space, exploration_level=0.1, random_state=None):
    """创建贝叶斯优化器，使用EI采集函数和Matern核"""
    kernel = Matern(length_scale=1.0, nu=2.5)
    # 高斯过程作为代理模型
    base_estimator = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=random_state
    )

    optimizer = BayesSearchCV(
        estimator=model_class(random_state=random_state),
        search_spaces=param_space,
        n_iter=30,  # 减少迭代次数以加快演示速度
        cv=3,
        n_jobs=-1,
        random_state=random_state,
        scoring='neg_mean_squared_error',
        optimizer_kwargs={
            'base_estimator': base_estimator,
            'acq_func': 'EI',
            'acq_func_kwargs': {'xi': exploration_level},
            'acq_optimizer': 'lbfgs',
            'n_initial_points': 5
        }
    )
    return optimizer


# ==================== 4. 模型训练与评估 ====================
def train_and_evaluate(model_class, model_name, param_space, X_train, y_train, X_test, y_test):
    """训练和评估单个模型"""
    print(f"\n=== 正在处理 {model_name} ===")

    # 贝叶斯优化
    optimizer = create_bayes_optimizer(model_class, param_space)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    optimizer.fit(X_train_scaled, y_train)

    # 获取最佳参数
    best_params = optimizer.best_params_
    print(f"{model_name}最佳参数: {best_params}")

    # 最终评估
    X_test_scaled = scaler.transform(X_test)
    final_model = model_class(**best_params, random_state=42)
    final_model.fit(X_train_scaled, y_train)
    test_pred = final_model.predict(X_test_scaled)
    test_score = mean_squared_error(y_test, test_pred)
    print(f"{model_name}测试集MSE: {test_score:.4f}")

    return {
        'model_name': model_name,
        'model': final_model,
        'best_params': best_params,
        'test_score': test_score,
        'optimizer': optimizer
    }


# ==================== 5. 学习曲线可视化 ====================
def plot_model_learning_curve(model, model_name, X_train, y_train, X_test, y_test):
    """绘制模型学习曲线"""
    plt.figure(figsize=(10, 6))

    train_errors, test_errors = [], []
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 确定评估点数量
    n_points = 20  # 固定评估点数量
    step_size = max(1, len(X_train) // n_points)
    evaluation_points = range(10, len(X_train), step_size)

    # 使用不同训练集大小评估性能
    for m in evaluation_points:
        model.fit(X_train_scaled[:m], y_train[:m])
        y_train_pred = model.predict(X_train_scaled[:m])
        y_test_pred = model.predict(X_test_scaled)
        train_errors.append(mean_squared_error(y_train[:m], y_train_pred))
        test_errors.append(mean_squared_error(y_test, y_test_pred))

    # 确保x和y维度匹配
    x_values = list(evaluation_points)
    if len(x_values) != len(train_errors):
        x_values = x_values[:len(train_errors)]  # 截断较长的部分

    plt.plot(x_values, train_errors, "r-+", linewidth=2, label="训练集")
    plt.plot(x_values, test_errors, "b-", linewidth=3, label="验证集")
    plt.xlabel("训练样本数", fontsize=14)
    plt.ylabel("MSE", fontsize=14)
    plt.title(f"{model_name}学习曲线", fontsize=14)
    plt.legend(loc="upper right", fontsize=14)
    plt.grid(True)

    # 确保输出目录存在
    os.makedirs("learning_curves", exist_ok=True)
    plt.savefig(f"learning_curves/{model_name}_learning_curve.png", dpi=300)
    plt.close()
    print(f"{model_name}学习曲线已保存")


# ==================== 6. 主执行流程 ====================
if __name__ == "__main__":
    # 数据准备
    file_path = "./数据/品位.xlsx"
    target_column = "Grade"
    X, y = load_and_preprocess_data(file_path, target_column)
    X_train, y_train, X_test, y_test = split_data_with_leakage_prevention(X, y, test_size=0.2)

    # 定义各模型参数空间（新增LGBM）
    models = [
        {
            'name': 'XGBoost',
            'class': XGBRegressor,
            'params': {
                'learning_rate': Real(0.01, 0.3, 'log-uniform'),
                'max_depth': Integer(3, 10),
                'n_estimators': Integer(100, 1000),
                'subsample': Real(0.6, 1.0),
                'colsample_bytree': Real(0.6, 1.0)
            }
        },
        {
            'name': 'GradientBoosting',
            'class': GradientBoostingRegressor,
            'params': {
                'learning_rate': Real(0.01, 0.2, 'log-uniform'),
                'n_estimators': Integer(50, 500),
                'max_depth': Integer(3, 8),
                'min_samples_split': Integer(2, 10),
                'min_samples_leaf': Integer(1, 5)
            }
        },
        {
            'name': 'RandomForest',
            'class': RandomForestRegressor,
            'params': {
                'n_estimators': Integer(50, 500),
                'max_depth': Integer(3, 15),
                'min_samples_split': Integer(2, 10),
                'min_samples_leaf': Integer(1, 5),
                'max_features': Categorical(['sqrt', 'log2', None])  # 移除了'auto'
            }
        },
        {
            'name': 'LightGBM',
            'class': LGBMRegressor,
            'params': {
                'learning_rate': Real(0.01, 0.3, 'log-uniform'),
                'num_leaves': Integer(20, 100),
                'max_depth': Integer(3, 12),
                'n_estimators': Integer(50, 1000),
                'min_child_samples': Integer(5, 50),
                'subsample': Real(0.6, 1.0),
                'colsample_bytree': Real(0.6, 1.0)
            }
        }
    ]

    # 训练和评估所有模型
    results = []
    for model_config in models:
        result = train_and_evaluate(
            model_config['class'],
            model_config['name'],
            model_config['params'],
            X_train, y_train,
            X_test, y_test
        )
        results.append(result)

        # 绘制学习曲线
        plot_model_learning_curve(
            result['model'],
            model_config['name'],
            X_train, y_train,
            X_test, y_test
        )

    # 打印最终比较结果
    print("\n=== 模型性能比较 ===")
    for result in results:
        print(f"{result['model_name']:15s} 测试MSE: {result['test_score']:.4f}")

    # 保存最佳模型
    best_model = min(results, key=lambda x: x['test_score'])
    print(f"\n最佳模型: {best_model['model_name']} (MSE: {best_model['test_score']:.4f})")