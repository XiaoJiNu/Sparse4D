#!/usr/bin/env python3
"""
测试类别定义和颜色映射
"""

import sys
import os
sys.path.append('.')

from mv2dfusion_visualizer import MV2DFusionVisualizer

def test_class_definitions():
    """测试类别定义和颜色映射"""
    print("=" * 60)
    print("测试百度数据类别定义和颜色映射")
    print("=" * 60)
    
    visualizer = MV2DFusionVisualizer()
    
    print(f"类别总数: {len(visualizer.CLASSES)}")
    print(f"颜色总数: {len(visualizer.ID_COLOR_MAP)}")
    
    print(f"\n类别定义:")
    print(f"{'ID':<3} {'类别名称':<25} {'颜色RGB'}")
    print("-" * 50)
    
    for i, (class_name, color) in enumerate(zip(visualizer.CLASSES, visualizer.ID_COLOR_MAP)):
        print(f"{i:<3} {class_name:<25} {color}")
    
    # 验证类别和颜色数量匹配
    if len(visualizer.CLASSES) == len(visualizer.ID_COLOR_MAP):
        print(f"\n✅ 类别和颜色数量匹配: {len(visualizer.CLASSES)} 个")
    else:
        print(f"\n❌ 类别和颜色数量不匹配:")
        print(f"   类别数量: {len(visualizer.CLASSES)}")
        print(f"   颜色数量: {len(visualizer.ID_COLOR_MAP)}")
    
    # 检查是否有重复的类别名称
    unique_classes = set(visualizer.CLASSES)
    if len(unique_classes) == len(visualizer.CLASSES):
        print(f"✅ 类别名称无重复")
    else:
        print(f"❌ 存在重复的类别名称")
        
    # 检查颜色值范围
    valid_colors = True
    for i, color in enumerate(visualizer.ID_COLOR_MAP):
        for j, channel in enumerate(color):
            if not (0 <= channel <= 255):
                print(f"❌ 颜色值超出范围 [0,255]: 类别{i} 通道{j} = {channel}")
                valid_colors = False
    
    if valid_colors:
        print(f"✅ 所有颜色值在有效范围内")
    
    print(f"\n预期的百度数据类别:")
    expected_classes = [
        'car', 'truck', 'bus', 'engineering_vehicle', 'cyclist', 'person', 
        'traffic_cone', 'tripod', 'tricycle', 'water_horse', 'pillar', 
        'noparking_pillar', 'gate_on', 'gate_off', 'anti_collision_bucket', 
        'noparking_board'
    ]
    
    print(f"预期类别数量: {len(expected_classes)}")
    print(f"实际类别数量: {len(visualizer.CLASSES)}")
    
    # 检查类别是否匹配
    missing_classes = set(expected_classes) - set(visualizer.CLASSES)
    extra_classes = set(visualizer.CLASSES) - set(expected_classes)
    
    if not missing_classes and not extra_classes:
        print(f"✅ 类别定义完全匹配预期")
    else:
        if missing_classes:
            print(f"❌ 缺少类别: {missing_classes}")
        if extra_classes:
            print(f"❌ 多余类别: {extra_classes}")
    
    return visualizer

def test_color_visualization():
    """测试颜色可视化效果（如果有matplotlib的话）"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        visualizer = MV2DFusionVisualizer()
        
        # 创建颜色条
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors_normalized = []
        for color in visualizer.ID_COLOR_MAP:
            # BGR to RGB conversion and normalize to [0,1]
            rgb_color = (color[2]/255, color[1]/255, color[0]/255)
            colors_normalized.append(rgb_color)
        
        # 绘制颜色条
        for i, (class_name, color) in enumerate(zip(visualizer.CLASSES, colors_normalized)):
            ax.barh(i, 1, color=color, edgecolor='black', linewidth=0.5)
            ax.text(0.5, i, f"{i}: {class_name}", ha='center', va='center', 
                   color='white' if sum(color) < 1.5 else 'black', fontweight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, len(visualizer.CLASSES) - 0.5)
        ax.set_yticks(range(len(visualizer.CLASSES)))
        ax.set_yticklabels([f"{i}" for i in range(len(visualizer.CLASSES))])
        ax.set_xlabel('类别颜色映射')
        ax.set_title('百度数据类别颜色映射表')
        
        plt.tight_layout()
        plt.savefig('./class_colors.png', dpi=150, bbox_inches='tight')
        print(f"\n✅ 颜色映射图已保存为 class_colors.png")
        
    except ImportError:
        print(f"\n⚠️  matplotlib不可用，跳过颜色可视化")

if __name__ == "__main__":
    visualizer = test_class_definitions()
    test_color_visualization()
    
    print(f"\n" + "=" * 60)
    print("类别定义测试完成")
    print("=" * 60)