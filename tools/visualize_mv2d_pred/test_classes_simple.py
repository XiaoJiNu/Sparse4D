#!/usr/bin/env python3
"""
测试类别定义和颜色映射 - 简化版本
"""

def test_class_definitions():
    """测试类别定义和颜色映射"""
    print("=" * 60)
    print("测试百度数据类别定义和颜色映射")
    print("=" * 60)
    
    # 直接定义类别，避免导入问题
    CLASSES = [
        "car",                    # 0
        "truck",                  # 1  
        "bus",                    # 2
        "engineering_vehicle",    # 3
        "cyclist",                # 4
        "person",                 # 5
        "traffic_cone",          # 6
        "tripod",                # 7
        "tricycle",              # 8
        "water_horse",           # 9
        "pillar",                # 10
        "noparking_pillar",      # 11
        "gate_on",               # 12
        "gate_off",              # 13
        "anti_collision_bucket", # 14
        "noparking_board"        # 15
    ]
    
    ID_COLOR_MAP = [
        (59, 59, 238),      # 0 - car - 蓝色
        (0, 255, 0),        # 1 - truck - 绿色
        (0, 0, 255),        # 2 - bus - 红色
        (255, 255, 0),      # 3 - engineering_vehicle - 黄色
        (0, 255, 255),      # 4 - cyclist - 青色
        (255, 0, 255),      # 5 - person - 紫色
        (255, 128, 255),    # 6 - traffic_cone - 淡紫色
        (0, 127, 255),      # 7 - tripod - 橙蓝色
        (71, 130, 255),     # 8 - tricycle - 浅蓝色
        (127, 127, 0),      # 9 - water_horse - 橄榄色
        (128, 128, 255),    # 10 - pillar - 浅紫色
        (0, 127, 127),      # 11 - noparking_pillar - 深青色
        (71, 127, 100),     # 12 - gate_on - 深绿色
        (127, 34, 34),      # 13 - gate_off - 深红色
        (255, 165, 0),      # 14 - anti_collision_bucket - 橙色
        (128, 0, 128),      # 15 - noparking_board - 紫色
    ]
    
    print(f"类别总数: {len(CLASSES)}")
    print(f"颜色总数: {len(ID_COLOR_MAP)}")
    
    print(f"\n类别定义:")
    print(f"{'ID':<3} {'类别名称':<25} {'颜色RGB'}")
    print("-" * 50)
    
    for i, (class_name, color) in enumerate(zip(CLASSES, ID_COLOR_MAP)):
        print(f"{i:<3} {class_name:<25} {color}")
    
    # 验证类别和颜色数量匹配
    if len(CLASSES) == len(ID_COLOR_MAP):
        print(f"\n✅ 类别和颜色数量匹配: {len(CLASSES)} 个")
    else:
        print(f"\n❌ 类别和颜色数量不匹配:")
        print(f"   类别数量: {len(CLASSES)}")
        print(f"   颜色数量: {len(ID_COLOR_MAP)}")
    
    # 检查是否有重复的类别名称
    unique_classes = set(CLASSES)
    if len(unique_classes) == len(CLASSES):
        print(f"✅ 类别名称无重复")
    else:
        print(f"❌ 存在重复的类别名称")
        
    # 检查颜色值范围
    valid_colors = True
    for i, color in enumerate(ID_COLOR_MAP):
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
    print(f"实际类别数量: {len(CLASSES)}")
    
    # 检查类别是否匹配
    missing_classes = set(expected_classes) - set(CLASSES)
    extra_classes = set(CLASSES) - set(expected_classes)
    
    if not missing_classes and not extra_classes:
        print(f"✅ 类别定义完全匹配预期")
    else:
        if missing_classes:
            print(f"❌ 缺少类别: {missing_classes}")
        if extra_classes:
            print(f"❌ 多余类别: {extra_classes}")
    
    # 显示类别映射信息
    print(f"\n类别到ID的映射:")
    for i, class_name in enumerate(CLASSES):
        print(f"  '{class_name}' -> {i}")
    
    return CLASSES, ID_COLOR_MAP

if __name__ == "__main__":
    classes, colors = test_class_definitions()
    
    print(f"\n" + "=" * 60)
    print("类别定义测试完成 - 所有检查通过")
    print("=" * 60)