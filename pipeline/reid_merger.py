"""
reid_merger.py
==============
사전 학습된 OSNet 모델을 사용해 트랙렛 특징 벡터를 추출하고,
동시성 제약 조건(Must-not-link Constraint)을 반영한 HAC(계층적 군집화)를 적용하여
서로 다른 카메라/시간대 간의 인물 ID를 병합하는 엔진.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from pipeline.tracklet_io import list_tracklets


class OSNetExtractor:
    """OSNet 특징 추출기 클래스."""

    def __init__(
        self,
        model_name: str = "osnet_x1_0",
        pretrained: bool = True,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[INFO] OSNet Extractor 디바이스 설정: {self.device}")

        # PyTorch Hub 또는 torchreid를 통해 OSNet 모델 로드
        self.model = self._load_model(model_name, pretrained)
        self.model.to(self.device)
        self.model.eval()

        # OSNet 공식 입력 스펙: Resize(256, 128) 및 ImageNet 정규화
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_model(self, model_name: str, pretrained: bool) -> nn.Module:
        # 1순위: torch.hub 로딩 시도 (deep-person-reid 공식 허브 리포지토리)
        try:
            print(f"[INFO] torch.hub를 통해 {model_name} 로딩 시도...")
            model = torch.hub.load(
                "KaiyangZhou/deep-person-reid",
                model_name,
                pretrained=pretrained
            )
            return model
        except Exception as e:
            print(f"[WARN] torch.hub 로딩 실패: {e}")

        # 2순위: torchreid 라이브러리가 로컬에 이미 설치된 경우 빌드 시도
        try:
            print(f"[INFO] 로컬 torchreid 모듈을 통해 {model_name} 로딩 시도...")
            import torchreid
            model = torchreid.models.build_model(
                name=model_name,
                num_classes=1000,
                pretrained=pretrained
            )
            return model
        except Exception as e:
            print(f"[WARN] torchreid 라이브러리 빌드 실패: {e}")

        # 3순위: 아키텍처와 가중치 파일 수동 다운로드 폴백 등 에러 메시지
        raise RuntimeError(
            f"OSNet 모델 ({model_name}) 로드 실패. "
            "인터넷 연결을 확인하거나 'pip install torchreid'를 수행하세요."
        )

    @torch.no_grad()
    def extract_image_feature(self, bgr_image: np.ndarray) -> np.ndarray:
        """단일 이미지(BGR)에서 L2 정규화된 512차원 특징 벡터 추출."""
        # BGR -> RGB 변환 및 PIL Image 생성
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_image)

        # 텐서 변환 및 배치 차원 추가
        img_t = self.transform(pil_img).unsqueeze(0).to(self.device)

        # 특징 추출 및 L2 정규화
        feat = self.model(img_t)  # shape: [1, 512]
        feat = feat / feat.norm(p=2, dim=1, keepdim=True)
        return feat.cpu().numpy()[0]

    @torch.no_grad()
    def extract_batch_features(self, bgr_images: List[np.ndarray], batch_size: int = 64) -> np.ndarray:
        """여러 이미지(BGR 리스트)에서 특징 벡터를 배치 단위로 효율적으로 추출."""
        features = []
        for i in range(0, len(bgr_images), batch_size):
            batch_imgs = bgr_images[i:i + batch_size]
            batch_tensors = []
            for bgr_img in batch_imgs:
                rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_img)
                batch_tensors.append(self.transform(pil_img))
            
            # shape: [B, 3, 256, 128]
            batch_t = torch.stack(batch_tensors).to(self.device)
            feat = self.model(batch_t)
            feat = feat / feat.norm(p=2, dim=1, keepdim=True)
            features.append(feat.cpu().numpy())
            
        return np.concatenate(features, axis=0)


class CrossCameraMerger:
    """특징 벡터 거리 계산 및 동시성 제약 HAC 클러스터링을 수행하는 병합 엔진."""

    def __init__(self, config: dict):
        self.config = config
        
        # 설정 로드
        reid_cfg = config.get("reid", {})
        self.model_name = reid_cfg.get("model_name", "osnet_x1_0")
        self.pretrained = reid_cfg.get("pretrained", True)
        self.device = reid_cfg.get("device", "auto")
        self.batch_size = reid_cfg.get("batch_size", 64)
        
        cluster_cfg = reid_cfg.get("clustering", {})
        self.threshold = cluster_cfg.get("threshold", 0.5)
        self.metric = cluster_cfg.get("metric", "cosine")
        self.linkage_method = cluster_cfg.get("linkage", "average")
        self.allow_cross_slot = cluster_cfg.get("allow_cross_slot", False)
        
        self.extractor = None

    def initialize_extractor(self):
        """특징 추출기 지연 초기화."""
        if self.extractor is None:
            self.extractor = OSNetExtractor(
                model_name=self.model_name,
                pretrained=self.pretrained,
                device=self.device
            )

    def extract_tracklet_representative_features(self, tracklets: List[Dict]) -> List[np.ndarray]:
        """
        각 트랙렛에 포함된 모든 프레임 크롭 특징 벡터를 평균(Average Pooling)하여
        대표 L2 정규화 특징 벡터를 생성.
        """
        self.initialize_extractor()
        print(f"[INFO] 총 {len(tracklets)}개 트랙렛의 특징 추출 시작...")

        # CPU/GPU 연산 속도 개선을 위해 트랙렛당 이미지 샘플링 수 제한 (최대 8장 균등 샘플링)
        # 모든 프레임을 다 쓰는 것과 유사도상 성능 차이가 거의 없으면서 연산 효율 극대화
        max_frames_per_tracklet = 8

        tracklet_features = []
        for t in tqdm(tracklets, desc="Extracting features"):
            tdir = Path(t["tracklet_dir"])
            crop_files = t.get("crop_files", [])
            
            if len(crop_files) > max_frames_per_tracklet:
                indices = np.linspace(0, len(crop_files) - 1, max_frames_per_tracklet, dtype=int)
                crop_files = [crop_files[idx] for idx in indices]

            # 트랙렛 폴더 안의 크롭 이미지 로드
            crops = []
            for fname in crop_files:
                fpath = tdir / fname
                if fpath.exists():
                    img = cv2.imread(str(fpath))
                    if img is not None and img.size > 0:
                        crops.append(img)
            
            if not crops:
                # 이미지 로드 불가 트랙렛 — None으로 마킹하여 클러스터링에서 제외
                tracklet_features.append(None)
                continue
                
            # 배치 단위 특징 추출
            feats = self.extractor.extract_batch_features(crops, batch_size=self.batch_size)
            
            # 각 프레임 특징 L2 정규화는 extract_batch_features 내부에서 수행됨.
            # 이들을 평균 낸 후 다시 최종 L2 정규화 적용.
            mean_feat = np.mean(feats, axis=0)
            norm = np.linalg.norm(mean_feat)
            if norm > 0:
                mean_feat = mean_feat / norm
            
            tracklet_features.append(mean_feat)
            
        return tracklet_features

    def filter_valid_tracklets(
        self, tracklets: List[Dict], features: List
    ) -> tuple:
        """None 특징 벡터(이미지 로드 불가) 트랙렛을 분리.

        Returns
        -------
        (valid_tracklets, valid_features, skipped_tracklets)
        """
        valid_t, valid_f, skipped = [], [], []
        for t, feat in zip(tracklets, features):
            if feat is None:
                skipped.append(t)
            else:
                valid_t.append(t)
                valid_f.append(feat)
        if skipped:
            print(f"[WARN] 이미지 로드 불가 트랙렛 {len(skipped)}개 → 클러스터링 제외, global_id = -1 마킹 예정")
        return valid_t, valid_f, skipped

    def mark_skipped_tracklets(self, skipped_tracklets: List[Dict]) -> None:
        """이미지 로드 불가 트랙렛의 metadata.json에 global_id = -1 기록."""
        for t in skipped_tracklets:
            t["global_id"] = -1
            meta_path = Path(t["tracklet_dir"]) / "metadata.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta["global_id"] = -1
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)
        if skipped_tracklets:
            print(f"[INFO] {len(skipped_tracklets)}개 트랙렛에 global_id = -1 마킹 완료")

    def _has_conflict(self, t1: Dict, t2: Dict) -> bool:
        """두 트랙렛이 must-not-link 관계인지 확인 (동일 cam+slot + 프레임 겹침)."""
        # 시간대 교차 병합 금지(over-merge 방지)
        if not self.allow_cross_slot and t1["time_slot"] != t2["time_slot"]:
            return True
        if t1["camera_id"] != t2["camera_id"] or t1["time_slot"] != t2["time_slot"]:
            return False
        f1_min, f1_max = min(t1["frame_indices"]), max(t1["frame_indices"])
        f2_min, f2_max = min(t2["frame_indices"]), max(t2["frame_indices"])
        return max(f1_min, f2_min) <= min(f1_max, f2_max)

    def enforce_must_not_link(self, labels: np.ndarray, tracklets: List[Dict]) -> np.ndarray:
        """HAC 결과에서 제약 위반 클러스터를 greedy split으로 강제 분리.

        각 클러스터 내 충돌 쌍을 찾아 그래프 컬러링으로 서브그룹 분리.
        첫 번째 서브그룹은 원래 레이블을 유지하고, 나머지는 새 레이블을 부여.
        """
        from collections import defaultdict

        cluster_to_idxs: dict = defaultdict(list)
        for idx, lbl in enumerate(labels):
            cluster_to_idxs[int(lbl)].append(idx)

        new_labels = labels.copy()
        next_label = int(max(labels)) + 1
        split_count = 0

        for lbl, idxs in cluster_to_idxs.items():
            # 클러스터 내 충돌 쌍 수집 (idxs 내 위치 기준)
            conflict_pairs: set = set()
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    if self._has_conflict(tracklets[idxs[i]], tracklets[idxs[j]]):
                        conflict_pairs.add((i, j))

            if not conflict_pairs:
                continue

            # Greedy coloring: 충돌 없는 첫 번째 서브그룹에 배정
            sub_groups: List[set] = []
            assignments: dict = {}

            for pos in range(len(idxs)):
                placed = False
                for sg_idx, sg in enumerate(sub_groups):
                    if not any(
                        (min(pos, p), max(pos, p)) in conflict_pairs for p in sg
                    ):
                        sg.add(pos)
                        assignments[pos] = sg_idx
                        placed = True
                        break
                if not placed:
                    sub_groups.append({pos})
                    assignments[pos] = len(sub_groups) - 1

            # 서브그룹 1+ 에 새 레이블 부여 (서브그룹 0은 원래 lbl 유지)
            for sg_idx in range(1, len(sub_groups)):
                new_lbl = next_label
                next_label += 1
                for pos in sub_groups[sg_idx]:
                    new_labels[idxs[pos]] = new_lbl
                    split_count += 1

        if split_count > 0:
            print(f"[INFO] 제약 위반 강제 분리: {split_count}개 트랙렛을 새 클러스터로 재배정")
        else:
            print("[INFO] 제약 위반 없음 — 추가 분리 불필요")
        return new_labels

    def compute_distance_matrix(self, features: List[np.ndarray]) -> np.ndarray:
        """특징 벡터 리스트 간 Cosine 거리 행렬 계산."""
        feats_arr = np.array(features)  # shape: [N, 512]
        
        # 모든 벡터가 L2 정규화되어 있으므로, 
        # Cosine Similarity = feats_arr @ feats_arr.T
        # Cosine Distance = 1.0 - Cosine Similarity
        sim_matrix = np.matmul(feats_arr, feats_arr.T)
        dist_matrix = 1.0 - sim_matrix
        
        # 수치적 오차 제거 (대각 성분 0, 거리 하한 0)
        np.fill_diagonal(dist_matrix, 0.0)
        dist_matrix = np.clip(dist_matrix, 0.0, 2.0)
        return dist_matrix

    def apply_must_not_link_constraints(self, dist_matrix: np.ndarray, tracklets: List[Dict]) -> np.ndarray:
        """
        Must-not-link Constraint (동시성 제약 조건)을 거리 행렬에 적용.
        동일한 카메라/시간대에서 시간축(프레임 번호)이 겹치는 트랙렛 쌍의 거리를 최댓값(999.0)으로 대체.
        """
        n = len(tracklets)
        constrained_matrix = dist_matrix.copy()
        
        constraint_count = 0
        for i in range(n):
            t1 = tracklets[i]
            cam1 = t1["camera_id"]
            slot1 = t1["time_slot"]
            f1_min, f1_max = min(t1["frame_indices"]), max(t1["frame_indices"])
            
            for j in range(i + 1, n):
                t2 = tracklets[j]
                # (신규) 시간대 교차 병합 금지 (over-merge 방지)
                if not self.allow_cross_slot and slot1 != t2["time_slot"]:
                    constrained_matrix[i, j] = 999.0
                    constrained_matrix[j, i] = 999.0
                    constraint_count += 1
                    continue
                # 동일 카메라 및 동일 슬롯인지 확인
                if cam1 == t2["camera_id"] and slot1 == t2["time_slot"]:
                    # 프레임 구간이 겹치는지 체크
                    f2_min, f2_max = min(t2["frame_indices"]), max(t2["frame_indices"])
                    
                    if max(f1_min, f2_min) <= min(f1_max, f2_max):
                        # 프레임 구간이 겹친다면 절대 병합될 수 없으므로 무한에 가까운 거리를 부여
                        constrained_matrix[i, j] = 999.0
                        constrained_matrix[j, i] = 999.0
                        constraint_count += 1
                        
        print(f"[INFO] 동시성 제약 조건(Must-not-link) 적용 완료: {constraint_count}개 쌍 감지 및 차단")
        return constrained_matrix

    def run_hac_clustering(self, dist_matrix: np.ndarray) -> np.ndarray:
        """HAC(계층적 군집화) 알고리즘을 수행하여 클러스터 레이블 반환."""
        # 1D condensed distance matrix 변환
        condensed_d = squareform(dist_matrix)
        
        # Linkage 수행
        z = linkage(condensed_d, method=self.linkage_method)
        
        # fcluster로 지정 임계값(threshold) 이하 병합
        labels = fcluster(z, t=self.threshold, criterion="distance")
        return labels

    def verify_clustering_results(self, labels: np.ndarray, tracklets: List[Dict]) -> bool:
        """제약 조건이 클러스터링 후 완벽히 지켜졌는지 검증."""
        cluster_to_indices = {}
        for idx, label in enumerate(labels):
            cluster_to_indices.setdefault(label, []).append(idx)
            
        success = True
        for label, idxs in cluster_to_indices.items():
            # 동일 클러스터 내의 모든 트랙렛 쌍에 대해 동시성 겹침 검사
            for i in range(len(idxs)):
                t1 = tracklets[idxs[i]]
                cam1 = t1["camera_id"]
                slot1 = t1["time_slot"]
                f1_min, f1_max = min(t1["frame_indices"]), max(t1["frame_indices"])
                
                for j in range(i + 1, len(idxs)):
                    t2 = tracklets[idxs[j]]
                    if cam1 == t2["camera_id"] and slot1 == t2["time_slot"]:
                        f2_min, f2_max = min(t2["frame_indices"]), max(t2["frame_indices"])
                        if max(f1_min, f2_min) <= min(f1_max, f2_max):
                            print(
                                f"[ERROR] 제약 조건 위반 발생! Global ID {label} 내부에서 "
                                f"동일 비디오(c{cam1}_t{slot1}) 겹침 트랙렛 병합됨: "
                                f"Track_{t1['track_id']:04d} (frame {f1_min}~{f1_max}) vs "
                                f"Track_{t2['track_id']:04d} (frame {f2_min}~{f2_max})"
                            )
                            success = False
        return success

    def update_tracklet_metadata_with_global_id(self, tracklets: List[Dict], labels: np.ndarray):
        """각 트랙렛의 metadata.json 파일에 global_id 업데이트 기록."""
        for t, label in zip(tracklets, labels):
            tdir = Path(t["tracklet_dir"])
            meta_path = tdir / "metadata.json"
            
            # 로컬 dict 업데이트
            t["global_id"] = int(label)
            
            # metadata.json 로드하여 업데이트 후 다시 저장
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                
                meta["global_id"] = int(label)
                
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)
