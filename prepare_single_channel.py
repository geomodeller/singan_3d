#!/usr/bin/env python3
"""
2채널 데이터를 단일 채널로 변환
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("단일 채널 3D 데이터 준비")
print("=" * 60)

try:
    # 첫 번째 샘플 로드
    data = np.load('first_sample_3d.npy')
    print(f"✅ 데이터 로드: {data.shape}")
    
    # 첫 번째 채널만 추출 (C, D, H, W) -> (D, H, W)
    if data.ndim == 4 and data.shape[0] == 2:
        # 첫 번째 채널 사용
        single_channel = data[0]  # shape: (16, 32, 32)
        print(f"✅ 첫 번째 채널 추출: {single_channel.shape}")
        print(f"✅ 데이터 범위: [{single_channel.min():.4f}, {single_channel.max():.4f}]")
        
        # 저장
        output_filename = 'single_channel_3d.npy'
        np.save(output_filename, single_channel)
        print(f"✅ 단일 채널 데이터 저장: {output_filename}")
        
        # 시각화
        mid_slice = single_channel.shape[0] // 2
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(single_channel[mid_slice, :, :], cmap='viridis')
        plt.title(f'Z-slice {mid_slice} (X-Y view)')
        plt.colorbar()
        
        plt.subplot(1, 3, 2)
        plt.imshow(single_channel[:, mid_slice, :], cmap='viridis')  
        plt.title(f'Y-slice {mid_slice} (X-Z view)')
        plt.colorbar()
        
        plt.subplot(1, 3, 3)
        plt.imshow(single_channel[:, :, mid_slice], cmap='viridis')
        plt.title(f'X-slice {mid_slice} (Y-Z view)')
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig('single_channel_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ 시각화 저장: single_channel_visualization.png")
        
        # 두 번째 채널도 확인 (비교용)
        second_channel = data[1]
        print(f"\n📊 두 번째 채널 정보: {second_channel.shape}")
        print(f"📊 두 번째 채널 범위: [{second_channel.min():.4f}, {second_channel.max():.4f}]")
        
        print("\n" + "=" * 60)
        print("단일 채널 데이터 준비 완료!")
        print("=" * 60)
        print(f"🎯 SinGAN 훈련용 파일: {output_filename}")
        print(f"📏 최종 데이터 크기: {single_channel.shape} (D×H×W)")
        
    else:
        print(f"❌ 예상과 다른 데이터 형태: {data.shape}")
        
except Exception as e:
    print(f"❌ 오류 발생: {e}") 