#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import yaml
import time
import logging
import threading
import queue
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import h5py
from PIL import Image

current_file = Path(__file__)
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))
from src.mtcnn_wrapper import FaceDetector
from src.va_emotion_core import VAEmotionCore


class LongVideoProcessor:
    """
    긴 영상 비디오 전처리기 (개선된 버전)
    얼굴 탐지 + VA 감정 특징 추출하여 원시 시퀀스 저장
    - 로그 개선: 명확한 통계 표시
    - 얼굴 저장 개선: 침착맨/다른사람 폴더 분리  
    - 배치 처리 개선: FaceNet 배치 크기 적용
    """
    
    def __init__(self, config_path: str = "video_analyzer/configs/inference_config.yaml"):
        """
        긴 영상 비디오 전처리기 초기화
        
        Args:
            config_path (str): 설정 파일 경로
        """
        # PyTorch 설정 최적화
        if torch.cuda.is_available():
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            try:
                if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                    torch.cuda.set_per_process_memory_fraction(0.7)
            except Exception as e:
                print(f"CUDA 최적화 설정 중 경고 (무시됨): {e}")
        
        # 설정 로드
        self.config = self._load_config(config_path)
        
        # 로깅 시스템 초기화
        self._setup_logging()
        
        # 개선된 성능 모니터링 변수
        self.stats = {
            'frames_processed': 0,           # 처리된 프레임 수
            'total_faces_detected': 0,       # MTCNN이 탐지한 전체 얼굴 수
            'chimchakman_faces': 0,          # 침착맨으로 인식된 얼굴 수
            'other_faces_filtered': 0,       # 다른 사람으로 필터링된 얼굴 수
            'emotions_extracted': 0,         # 감정이 추출된 얼굴 수
            'processing_start_time': None,
            'last_stats_time': time.time(),
            'batch_count': 0,
            'total_face_detection_time': 0,
            'total_emotion_time': 0,
            'total_recognition_time': 0,
            'others_faces_saved': 0          # 저장된 다른 사람 얼굴 수
        }
        
        # 종료 플래그
        self.stop_flag = False
        self.face_detection_done = False
        
        # 큐 생성
        self.frame_queue = queue.Queue(maxsize=self.config['performance']['max_queue_size'])
        self.result_queue = queue.Queue()
        
        # 모델들 초기화
        self._init_face_detector()
        self._init_emotion_model()
        self._init_face_recognition()
        
        # 기본 출력 디렉토리 생성
        self._create_base_output_dirs()
        
        self.logger.info("✅ 긴 영상 비디오 전처리기 초기화 완료 (개선된 버전)")
        self._print_config_summary()
    
    def _load_config(self, config_path: str) -> Dict:
        """설정 파일 로드"""
        default_config = {
            'video': {
                'frame_skip': 15,
                'extract_emotions': True,
                'save_face_images': True,
                'face_images_dir': 'debug_faces',
                'save_others_faces': True,
                'max_others_faces': 200
            },
            'mtcnn': {
                'batch_size': 32,
                'image_size': 224,
                'margin': 20,
                'prob_threshold': 0.9,
                'align_faces': False
            },
            'emotion': {
                'model_path': 'models/affectnet_emotions/enet_b0_8_va_mtl.pt',
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'batch_size': 32
            },
            'face_recognition': {
                'enabled': True,
                'test_mode': False,
                'embedding_path': 'face_recognition/target_embeddings/chimchakman.npy',
                'similarity_threshold': 0.7,
                'batch_size': 32,
                'logging': {
                    'save_filtered_faces': True,
                    'log_filtered_count': True,
                    'save_similarity_info': True
                }
            },
            'output': {
                'base_dir': 'video_analyzer',
                'preprocessed_dir': 'preprocessed_data',
                'video_sequence_dir': 'video_sequences'
            },
            'performance': {
                'max_queue_size': 200,
                'monitoring_interval': 10.0
            },
            'logging': {
                'level': 'INFO',
                'batch_summary': True,
                'save_face_statistics': True
            },
            'debug': {
                'clear_previous_faces': True,
                'face_filename_format': 'timestamp_{timestamp:06d}_face{face_idx}_{type}_sim{similarity:.3f}.jpg'
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            # 기본값과 병합
            def merge_dict(default, loaded):
                for key, value in default.items():
                    if key not in loaded:
                        loaded[key] = value
                    elif isinstance(value, dict) and isinstance(loaded[key], dict):
                        merge_dict(value, loaded[key])
                return loaded
            config = merge_dict(default_config, config)
        else:
            config = default_config
            print(f"⚠️ 설정 파일 없음, 기본값 사용: {config_path}")
        
        return config
    
    def _setup_logging(self):
        """로깅 시스템 설정"""
        self.logger = logging.getLogger('LongVideoProcessor')
        self.logger.setLevel(getattr(logging, self.config['logging']['level']))
        
        # 기존 핸들러 제거
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def _init_face_detector(self):
        """MTCNN 얼굴 탐지기 초기화"""
        mtcnn_config = self.config['mtcnn']
        
        self.face_detector = FaceDetector(
            image_size=mtcnn_config['image_size'],
            margin=mtcnn_config['margin'],
            prob_threshold=mtcnn_config['prob_threshold'],
            align_faces=mtcnn_config['align_faces']
        )
        
        self.logger.info(f"✅ MTCNN 초기화 완료 (배치 크기: {mtcnn_config['batch_size']})")
    
    def _init_emotion_model(self):
        """VA 감정 모델 초기화"""
        emotion_config = self.config['emotion']
        
        try:
            self.emotion_model = VAEmotionCore(
                model_path=emotion_config['model_path'],
                device=emotion_config['device']
            )
            self.logger.info("✅ VA 감정 모델 로드 완료")
        except Exception as e:
            self.logger.error(f"❌ 감정 모델 로드 실패: {e}")
            raise
    
    def _init_face_recognition(self):
        """FaceNet 얼굴 인식 모델 초기화"""
        face_recog_config = self.config['face_recognition']
        
        # 얼굴 인식이 비활성화되었거나 테스트 모드인 경우
        if not face_recog_config.get('enabled', False) or face_recog_config.get('test_mode', False):
            mode_str = "비활성화됨" if not face_recog_config.get('enabled', False) else "테스트 모드"
            self.logger.info(f"⚠️ 얼굴 인식 {mode_str}")
            self.facenet_model = None
            self.target_embedding = None
            return
        
        try:
            from facenet_pytorch import InceptionResnetV1
            
            # FaceNet을 CPU에 로드 (안정성)
            cpu_device = torch.device('cpu')
            
            with torch.no_grad():
                self.facenet_model = InceptionResnetV1(pretrained='vggface2')
                self.facenet_model = self.facenet_model.to(cpu_device).eval()
            
            # 타겟 임베딩 로드
            embedding_path = face_recog_config['embedding_path']
            if os.path.exists(embedding_path):
                embedding_data = np.load(embedding_path, allow_pickle=True).item()
                self.target_embedding = torch.tensor(embedding_data['embedding']).to(cpu_device)
                
                self.logger.info(f"✅ 얼굴 인식 초기화 완료")
                self.logger.info(f"   임베딩 파일: {embedding_path}")
                self.logger.info(f"   유사도 임계값: {face_recog_config['similarity_threshold']}")
                self.logger.info(f"   배치 크기: {face_recog_config['batch_size']}")
            else:
                self.logger.error(f"❌ 임베딩 파일을 찾을 수 없습니다: {embedding_path}")
                self.facenet_model = None
                self.target_embedding = None
            
        except Exception as e:
            self.logger.error(f"❌ 얼굴 인식 초기화 실패: {e}")
            self.facenet_model = None
            self.target_embedding = None
    

    def _create_base_output_dirs(self):
        """기본 출력 디렉토리 생성"""
        base_dir = self.config['output']['base_dir']
        self.preprocessed_dir = os.path.join(base_dir, self.config['output']['preprocessed_dir'])
        self.video_sequence_dir = os.path.join(self.preprocessed_dir, self.config['output']['video_sequence_dir'])
        
        os.makedirs(self.video_sequence_dir, exist_ok=True)

    def _create_video_specific_dirs(self, video_name: str):
        """영상별 얼굴 이미지 디렉토리 생성"""
        if not self.config['video']['save_face_images']:
            return
        
        # 날짜 추가
        date_str = datetime.now().strftime("%Y%m%d")
        
        # 안전한 비디오 이름 생성 (특수문자 제거)
        safe_video_name = "".join(c for c in video_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_video_name = safe_video_name.replace(' ', '_')
        
        # 영상별 폴더명 생성
        video_folder_name = f"chimchakman_{safe_video_name}_{date_str}"
        
        # 얼굴 이미지 디렉토리 생성
        self.face_images_dir = os.path.join(
            self.preprocessed_dir, 
            self.config['video']['face_images_dir'],
            video_folder_name
        )
        
        # 하위 폴더 생성
        self.chimchakman_faces_dir = os.path.join(self.face_images_dir, "chimchakman")
        self.others_faces_dir = os.path.join(self.face_images_dir, "others")
        
        os.makedirs(self.chimchakman_faces_dir, exist_ok=True)
        os.makedirs(self.others_faces_dir, exist_ok=True)
        
        self.logger.info(f"📁 영상별 얼굴 폴더 생성: {video_folder_name}")
        
        # 기존 파일들 정리 (옵션)
        if self.config['debug']['clear_previous_faces']:
            self._clear_debug_faces()
    
    def _clear_debug_faces(self):
        """이전 디버그 얼굴 이미지들 정리"""
        try:
            for folder in [self.chimchakman_faces_dir, self.others_faces_dir]:
                if os.path.exists(folder):
                    for file in os.listdir(folder):
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            os.remove(os.path.join(folder, file))
            self.logger.info("🧹 이전 디버그 얼굴 이미지들 정리 완료")
        except Exception as e:
            self.logger.warning(f"⚠️ 디버그 이미지 정리 실패: {e}")
    
    def _print_config_summary(self):
        """설정 요약 출력"""
        self.logger.info("📋 설정 요약:")
        self.logger.info(f"   프레임 스킵: {self.config['video']['frame_skip']}프레임마다")
        self.logger.info(f"   MTCNN 배치: {self.config['mtcnn']['batch_size']}")
        self.logger.info(f"   감정 배치: {self.config['emotion']['batch_size']}")
        
        # 얼굴 인식 설정 출력
        if self.config['face_recognition']['enabled']:
            if self.facenet_model is not None:
                self.logger.info(f"   얼굴 인식: 활성화 (임계값: {self.config['face_recognition']['similarity_threshold']}, 배치: {self.config['face_recognition']['batch_size']})")
            else:
                self.logger.info(f"   얼굴 인식: 설정됨 (모델 로드 실패)")
        else:
            self.logger.info(f"   얼굴 인식: 비활성화")
        
        self.logger.info(f"   감정 추출: {self.config['video']['extract_emotions']}")
        self.logger.info(f"   얼굴 이미지 저장: {self.config['video']['save_face_images']}")
        if self.config['video']['save_face_images']:
            self.logger.info(f"   다른 사람 얼굴 저장: {self.config['video']['save_others_faces']} (최대 {self.config['video']['max_others_faces']}개)")
        self.logger.info(f"   디바이스: {self.config['emotion']['device']}")

    def process_long_video(self, video_path: str) -> Optional[Dict]:
            """
            긴 영상 전처리
            
            Args:
                video_path (str): 처리할 비디오 파일 경로
                
            Returns:
                Dict: 전처리 결과 정보
            """
            if not os.path.exists(video_path):
                self.logger.error(f"❌ 비디오 파일을 찾을 수 없습니다: {video_path}")
                return None
            
            video_name = Path(video_path).stem
            self.logger.info(f"🎬 긴 영상 전처리 시작: {video_name}")
            
            # 비디오별 출력 디렉토리 생성 (필요한 경우)
            if self.config['video']['save_face_images']:
                self._create_video_specific_dirs(video_name)
            
            # 비디오 정보 가져오기
            video_info = self._get_video_info(video_path)
            self.logger.info(f"   길이: {video_info['duration']:.1f}초, FPS: {video_info['fps']:.1f}")
            
            # 결과 저장용 초기화
            self.emotion_sequences = []
            self.face_detected_sequence = []
            self.timestamps_sequence = []
            self.frame_indices_sequence = []
            
            self.stats['processing_start_time'] = time.time()
            self.face_detection_done = False
            self.stop_flag = False
            
            # 스레드 시작
            threads = []
            
            # 1. 프레임 읽기 스레드
            frame_thread = threading.Thread(target=self._frame_reader_worker, args=(video_path,))
            threads.append(frame_thread)
            
            # 2. 얼굴 탐지 + 감정 추출 스레드
            process_thread = threading.Thread(target=self._face_emotion_worker)
            threads.append(process_thread)
            
            # 3. 성능 모니터링 스레드
            monitor_thread = threading.Thread(target=self._performance_monitor)
            monitor_thread.daemon = True
            threads.append(monitor_thread)
            
            # 모든 스레드 시작
            for thread in threads:
                thread.start()
            
            # 프레임 읽기 완료 대기
            frame_thread.join()
            self.logger.info("✅ 프레임 읽기 완료")
            
            # 처리 완료 대기
            process_thread.join()
            self.logger.info("✅ 얼굴 탐지 및 감정 추출 완료")
            
            # 종료 플래그 설정
            self.stop_flag = True
            
            # 결과 저장
            result = self._save_results(video_path, video_info)
            
            # 얼굴 통계 파일 저장
            if self.config['logging']['save_face_statistics']:
                self._save_face_statistics()
            
            # 최종 통계 출력
            self._print_final_stats()
            
            return result
        
    def _get_video_info(self, video_path: str) -> Dict:
        """비디오 정보 추출"""
        cap = cv2.VideoCapture(video_path)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration
        }

    def _frame_reader_worker(self, video_path: str):
        """프레임 읽기 워커"""
        cap = cv2.VideoCapture(video_path)
        frame_skip = self.config['video']['frame_skip']
        frame_count = 0
        
        # 호환성 확인
        if not cap.isOpened():
            self.logger.error("❌ 비디오 파일 열기 실패")
            self.frame_queue.put(None)
            return
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 프레임 스킵
                if frame_count % frame_skip != 0:
                    frame_count += 1
                    continue
                
                # BGR to RGB 변환
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 타임스탬프 계산
                timestamp = frame_count / cap.get(cv2.CAP_PROP_FPS)
                
                # 큐에 추가
                frame_data = {
                    'frame': frame_rgb,
                    'frame_number': frame_count,
                    'timestamp': timestamp
                }
                
                self.frame_queue.put(frame_data)
                frame_count += 1
                
        except Exception as e:
            self.logger.error(f"❌ 프레임 읽기 오류: {e}")
        finally:
            cap.release()
            # 종료 신호
            self.frame_queue.put(None)

    def _face_emotion_worker(self):
        """얼굴 탐지 + 감정 추출 워커 (배치 최적화)"""
        batch_size = self.config['mtcnn']['batch_size']
        frame_batch = []
        
        try:
            while True:
                frame_data = self.frame_queue.get()
                
                if frame_data is None:  # 종료 신호
                    # 남은 배치 처리
                    if frame_batch:
                        self._process_frame_batch_optimized(frame_batch)
                    break
                
                frame_batch.append(frame_data)
                
                # 배치가 찼으면 처리
                if len(frame_batch) >= batch_size:
                    self._process_frame_batch_optimized(frame_batch)
                    frame_batch = []
                
                self.frame_queue.task_done()
                
        except Exception as e:
            self.logger.error(f"❌ 얼굴 탐지 및 감정 추출 오류: {e}")
        finally:
            self.face_detection_done = True

    def _process_frame_batch_optimized(self, frame_batch: List[Dict]):
        """프레임 배치에서 얼굴 탐지 및 감정 추출 (최적화)"""
        batch_start_time = time.time()
        
        try:
            # 1. MTCNN 배치 처리
            pil_images = []
            frame_metadata_list = []
            
            for frame_data in frame_batch:
                pil_image = Image.fromarray(frame_data['frame'])
                pil_images.append(pil_image)
                frame_metadata_list.append({
                    'frame_number': frame_data['frame_number'],
                    'timestamp': frame_data['timestamp']
                })
            
            # 얼굴 탐지
            detection_start = time.time()
            face_results = self.face_detector.process_image_batch(pil_images, frame_metadata_list)
            self.stats['total_face_detection_time'] += time.time() - detection_start
            
            # 탐지된 얼굴 수 업데이트
            self.stats['total_faces_detected'] += len(face_results)
            
            # 2. 얼굴 인식 필터링 (배치) - 개선된 버전
            if self.config['face_recognition']['enabled'] and self.facenet_model is not None:
                recognition_start = time.time()
                face_results = self._filter_faces_by_recognition_batch(face_results)
                self.stats['total_recognition_time'] += time.time() - recognition_start
            
            # 3. 감정 추출 (배치)
            if self.config['video']['extract_emotions'] and face_results:
                emotion_start = time.time()
                self._extract_emotions_batch(face_results)
                self.stats['total_emotion_time'] += time.time() - emotion_start
            
            # 4. 프레임별 결과 매칭 및 저장
            self._match_and_store_results(frame_batch, face_results)
            
            # 통계 업데이트
            self.stats['batch_count'] += 1
            batch_time = time.time() - batch_start_time
            
            if self.config['logging']['batch_summary']:
                faces_count = len(face_results)
                self.logger.debug(
                    f"배치 처리: {len(frame_batch)}프레임 → {faces_count}개 얼굴 ({batch_time:.2f}초)"
                )
            
        except Exception as e:
            self.logger.error(f"❌ 배치 처리 실패: {e}")

    def _filter_faces_by_recognition_batch(self, face_results: List[Dict]) -> List[Dict]:
        """얼굴 인식으로 침착맨 얼굴만 필터링 (개선된 배치 처리)"""
        if not face_results:
            return face_results
        
        try:
            # 얼굴 이미지들 추출
            face_images = [result['face_image'] for result in face_results]
            
            # 개선된 배치 임베딩 추출 (배치 크기 적용)
            embeddings = self._get_face_embeddings_batch(face_images)
            if embeddings is None:
                return face_results
            
            # 배치 유사도 계산
            similarities = self._calculate_similarities_batch(embeddings)
            if similarities is None:
                return face_results
            
            # 임계값 기준으로 필터링
            threshold = self.config['face_recognition']['similarity_threshold']
            matches = similarities > threshold
            
            # 필터링된 결과 생성
            filtered_results = []
            for i, (result, similarity, is_match) in enumerate(zip(face_results, similarities, matches)):
                if is_match:
                    # 침착맨 얼굴
                    result['similarity'] = float(similarity)
                    filtered_results.append(result)
                    self.stats['chimchakman_faces'] += 1
                    
                    # 침착맨 폴더에 저장
                    if self.config['video']['save_face_images']:
                        self._save_chimchakman_face(result, float(similarity))
                else:
                    # 다른 사람 얼굴
                    self.stats['other_faces_filtered'] += 1
                    
                    # 다른 사람 폴더에 저장 (개수 제한)
                    if (self.config['video']['save_face_images'] and 
                        self.config['video']['save_others_faces'] and
                        self.stats['others_faces_saved'] < self.config['video']['max_others_faces']):
                        self._save_other_face(result, float(similarity))
                        self.stats['others_faces_saved'] += 1
            
            # 필터링 통계 로깅
            if self.config['face_recognition']['logging']['log_filtered_count']:
                self.logger.debug(
                    f"얼굴 인식: {len(face_results)}개 → {len(filtered_results)}개 "
                    f"(필터링: {len(face_results) - len(filtered_results)}개)"
                )
            
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"❌ 얼굴 인식 필터링 실패: {e}")
            return face_results

    def _get_face_embeddings_batch(self, face_images: List[Image.Image]) -> Optional[torch.Tensor]:
        """FaceNet 배치 임베딩 추출 (개선된 버전 - 배치 크기 적용)"""
        if self.facenet_model is None or not face_images:
            return None
        
        try:
            batch_size = self.config['face_recognition']['batch_size']
            all_embeddings = []
            
            # 배치 크기별로 나누어 처리
            for i in range(0, len(face_images), batch_size):
                batch_end = min(i + batch_size, len(face_images))
                batch_images = face_images[i:batch_end]
                
                # 160x160 리사이징 (FaceNet 입력 크기)
                resized_images = []
                for face_img in batch_images:
                    resized_img = face_img.resize((160, 160), Image.BILINEAR)
                    img_array = np.array(resized_img)
                    img_tensor = torch.from_numpy(img_array).float().permute(2, 0, 1)
                    img_tensor = (img_tensor - 127.5) / 128.0  # 정규화 [-1, 1]
                    resized_images.append(img_tensor)
                
                # 배치 텐서 생성
                batch_tensor = torch.stack(resized_images).to(self.facenet_model.device)
                
                # 임베딩 추출
                with torch.no_grad():
                    batch_embeddings = self.facenet_model(batch_tensor)
                    batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                    all_embeddings.append(batch_embeddings)
            
            # 모든 배치 결과 연결
            return torch.cat(all_embeddings, dim=0)
            
        except Exception as e:
            self.logger.error(f"❌ 배치 임베딩 추출 실패: {e}")
            return None

            

    def _calculate_similarities_batch(self, embeddings: torch.Tensor) -> Optional[torch.Tensor]:
        """배치 임베딩과 타겟 임베딩 간의 유사도 계산"""
        if embeddings is None or self.target_embedding is None:
            return None
        
        try:
            # 계산을 위해 임베딩과 타겟이 같은 디바이스에 있어야 함
            device = embeddings.device
            target_embedding = self.target_embedding.to(device)
            
            # 타겟 임베딩을 배치 크기로 확장
            target_expanded = target_embedding.unsqueeze(0).repeat(embeddings.size(0), 1)
            
            # 코사인 유사도 계산
            similarities = F.cosine_similarity(embeddings, target_expanded)
            
            return similarities
            
        except Exception as e:
            self.logger.error(f"❌ 유사도 계산 실패: {e}")
            return None
    
    def _extract_emotions_batch(self, face_results: List[Dict]):
        """감정 추출 (배치 처리)"""
        if not face_results:
            return
        
        try:
            emotion_batch_size = self.config['emotion']['batch_size']
            
            # 배치 단위로 처리
            for i in range(0, len(face_results), emotion_batch_size):
                batch_end = min(i + emotion_batch_size, len(face_results))
                batch_faces = face_results[i:batch_end]
                
                # 배치 얼굴 이미지 추출
                face_images = [result['face_image'] for result in batch_faces]
                
                # 배치 감정 추출
                emotion_features = self.emotion_model.extract_emotion_features_batch(face_images)
                
                # 결과를 face_results에 저장
                for j, features in enumerate(emotion_features):
                    batch_faces[j]['emotion_features'] = features
                    self.stats['emotions_extracted'] += 1
                    
        except Exception as e:
            self.logger.error(f"❌ 배치 감정 추출 실패: {e}")
    
    def _match_and_store_results(self, frame_batch: List[Dict], face_results: List[Dict]):
        """프레임별 결과 매칭 및 저장"""
        # 프레임별로 처리
        for frame_data in frame_batch:
            frame_number = frame_data['frame_number']
            timestamp = frame_data['timestamp']
            
            # 해당 프레임의 얼굴 찾기
            frame_faces = [f for f in face_results if f['frame_number'] == frame_number]
            
            if frame_faces:
                # 얼굴이 있는 경우 - 가장 높은 유사도의 얼굴 선택
                if len(frame_faces) > 1 and 'similarity' in frame_faces[0]:
                    face_data = max(frame_faces, key=lambda x: x.get('similarity', 0))
                else:
                    face_data = frame_faces[0]
                
                # 감정 특징 가져오기
                if 'emotion_features' in face_data:
                    emotion_features = face_data['emotion_features']
                else:
                    emotion_features = np.zeros(10)
                
                # 결과 저장
                self.emotion_sequences.append(emotion_features)
                self.face_detected_sequence.append(True)
                
            else:
                # 얼굴이 없는 경우
                self.emotion_sequences.append(np.full(10, np.nan))
                self.face_detected_sequence.append(False)
            
            # 공통 정보 저장
            self.timestamps_sequence.append(timestamp)
            self.frame_indices_sequence.append(frame_number)
            self.stats['frames_processed'] += 1
    
    def _save_chimchakman_face(self, face_data: Dict, similarity: float):
        """침착맨 얼굴 저장 (타임스탬프 파일명)"""
        try:
            timestamp = face_data.get('timestamp', 0)
            face_idx = face_data.get('face_index', 0)
            
            # 타임스탬프를 밀리초 단위 정수로 변환 (15.750초 → 015750)
            timestamp_ms = int(timestamp * 1000)
            
            # 타임스탬프 파일명 형식 사용
            filename_format = self.config['debug']['face_filename_format']
            filename = filename_format.format(
                timestamp=timestamp_ms,
                face_idx=face_idx,
                type="chimchakman",
                similarity=similarity
            )
            
            save_path = os.path.join(self.chimchakman_faces_dir, filename)
            face_data['face_image'].save(save_path)
            
        except Exception as e:
            self.logger.warning(f"⚠️ 침착맨 얼굴 저장 실패: {e}")
    
    def _save_other_face(self, face_data: Dict, similarity: float):
        """다른 사람 얼굴 저장 (타임스탬프 파일명)"""
        try:
            timestamp = face_data.get('timestamp', 0)
            face_idx = face_data.get('face_index', 0)
            
            # 타임스탬프를 밀리초 단위 정수로 변환
            timestamp_ms = int(timestamp * 1000)
            
            # 타임스탬프 파일명 형식 사용
            filename_format = self.config['debug']['face_filename_format']
            filename = filename_format.format(
                timestamp=timestamp_ms,
                face_idx=face_idx,
                type="other",
                similarity=similarity
            )
            
            save_path = os.path.join(self.others_faces_dir, filename)
            face_data['face_image'].save(save_path)
            
        except Exception as e:
            self.logger.warning(f"⚠️ 다른 사람 얼굴 저장 실패: {e}")
    
    def _save_face_statistics(self):
        """얼굴 인식 통계를 텍스트 파일로 저장"""
        try:
            if not hasattr(self, 'face_images_dir'):
                return
                
            stats_path = os.path.join(self.face_images_dir, "face_statistics.txt")
            
            total_faces = self.stats['chimchakman_faces'] + self.stats['other_faces_filtered']
            chimchakman_rate = self.stats['chimchakman_faces'] / total_faces * 100 if total_faces > 0 else 0
            frame_face_rate = self.stats['chimchakman_faces'] / max(1, self.stats['frames_processed']) * 100
            
            with open(stats_path, 'w', encoding='utf-8') as f:
                f.write(f"얼굴 인식 통계 보고서\n")
                f.write(f"{'='*50}\n")
                f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write(f"📊 처리 통계\n")
                f.write(f"├─ 처리된 프레임: {self.stats['frames_processed']:,}개\n")
                f.write(f"├─ 전체 탐지된 얼굴: {total_faces:,}개\n")
                f.write(f"├─ 침착맨 얼굴: {self.stats['chimchakman_faces']:,}개 ({chimchakman_rate:.1f}%)\n")
                f.write(f"├─ 다른 사람 얼굴: {self.stats['other_faces_filtered']:,}개\n")
                f.write(f"└─ 감정 분석 완료: {self.stats['emotions_extracted']:,}개\n\n")
                
                f.write(f"📁 저장된 파일\n")
                f.write(f"├─ 침착맨 얼굴: {self.stats['chimchakman_faces']:,}개\n")
                f.write(f"└─ 다른 사람 얼굴: {min(self.stats['others_faces_saved'], self.config['video']['max_others_faces']):,}개 (최대 {self.config['video']['max_others_faces']}개 제한)\n\n")
                
                f.write(f"🎯 성능 지표\n")
                f.write(f"├─ 프레임당 침착맨 비율: {frame_face_rate:.1f}%\n")
                f.write(f"├─ 평균 탐지 시간: {self.stats['total_face_detection_time']/max(1, self.stats['batch_count']):.3f}초/배치\n")
                f.write(f"├─ 평균 인식 시간: {self.stats['total_recognition_time']/max(1, self.stats['batch_count']):.3f}초/배치\n")
                f.write(f"└─ 평균 감정 시간: {self.stats['total_emotion_time']/max(1, self.stats['batch_count']):.3f}초/배치\n")
            
            self.logger.info(f"📄 얼굴 통계 파일 저장: {stats_path}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 통계 파일 저장 실패: {e}")
    
    def _save_results(self, video_path: str, video_info: Dict) -> Dict:
        """결과를 HDF5 파일로 저장"""
        try:
            video_name = Path(video_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            hdf5_filename = f"video_seq_{video_name}_{timestamp}.h5"
            hdf5_path = os.path.join(self.video_sequence_dir, hdf5_filename)
            
            with h5py.File(hdf5_path, 'w') as f:
                # 메타데이터
                f.attrs['video_name'] = video_name
                f.attrs['video_path'] = video_path
                f.attrs['duration'] = video_info['duration']
                f.attrs['fps'] = video_info['fps']
                f.attrs['frame_skip'] = self.config['video']['frame_skip']
                f.attrs['total_frames'] = len(self.emotion_sequences)
                f.attrs['face_detection_ratio'] = float(np.mean(self.face_detected_sequence))
                f.attrs['processed_at'] = datetime.now().isoformat()
                
                # 🆕 얼굴 이미지 폴더 경로 추가
                if hasattr(self, 'face_images_dir') and self.config['video']['save_face_images']:
                    # 상대 경로로 저장 (프로젝트 루트 기준)
                    relative_face_dir = os.path.relpath(self.face_images_dir)
                    f.attrs['face_images_dir'] = relative_face_dir
                    f.attrs['chimchakman_faces_dir'] = os.path.relpath(self.chimchakman_faces_dir)
                    f.attrs['others_faces_dir'] = os.path.relpath(self.others_faces_dir)
                    self.logger.info(f"💾 HDF5에 얼굴 폴더 경로 저장: {relative_face_dir}")
                
                # 개선된 얼굴 인식 통계
                total_faces = self.stats['chimchakman_faces'] + self.stats['other_faces_filtered']
                f.attrs['total_faces_detected'] = total_faces
                f.attrs['chimchakman_faces'] = self.stats['chimchakman_faces']
                f.attrs['other_faces_filtered'] = self.stats['other_faces_filtered']
                if total_faces > 0:
                    f.attrs['chimchakman_recognition_rate'] = float(self.stats['chimchakman_faces'] / total_faces)
                else:
                    f.attrs['chimchakman_recognition_rate'] = 0.0
                
                # 시퀀스 데이터
                sequences_group = f.create_group('sequences')
                sequences_group.create_dataset('emotions', 
                                             data=np.array(self.emotion_sequences),
                                             compression='gzip')
                sequences_group.create_dataset('face_detected', 
                                             data=np.array(self.face_detected_sequence),
                                             compression='gzip')
                sequences_group.create_dataset('timestamps', 
                                             data=np.array(self.timestamps_sequence),
                                             compression='gzip')
                sequences_group.create_dataset('frame_indices', 
                                             data=np.array(self.frame_indices_sequence),
                                             compression='gzip')
            
            # 개선된 결과 반환
            result = {
                'video_name': video_name,
                'video_path': video_path,
                'hdf5_path': hdf5_path,
                'duration': video_info['duration'],
                'total_frames': len(self.emotion_sequences),
                'face_detection_ratio': float(np.mean(self.face_detected_sequence)),
                'chimchakman_recognition_rate': float(self.stats['chimchakman_faces'] / max(1, total_faces)),
                'face_images_dir': getattr(self, 'face_images_dir', None),
                'stats': self.stats.copy()
            }
            
            self.logger.info(f"💾 결과 저장: {hdf5_path}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 결과 저장 실패: {e}")
            return None
    
    def _performance_monitor(self):
        """성능 모니터링 (개선된 로그)"""
        interval = self.config['performance']['monitoring_interval']
        
        while not self.stop_flag:
            time.sleep(interval)
            
            if self.stats['processing_start_time'] is None:
                continue
            
            # 현재 통계
            elapsed = time.time() - self.stats['processing_start_time']
            fps = self.stats['frames_processed'] / elapsed if elapsed > 0 else 0
            
            total_faces = self.stats['chimchakman_faces'] + self.stats['other_faces_filtered']
            chimchakman_rate = (self.stats['chimchakman_faces'] / max(1, total_faces)) * 100
            frame_face_rate = (self.stats['chimchakman_faces'] / max(1, self.stats['frames_processed'])) * 100
            
            # 평균 처리 시간들
            avg_detection = self.stats['total_face_detection_time'] / max(1, self.stats['batch_count'])
            avg_emotion = self.stats['total_emotion_time'] / max(1, self.stats['batch_count'])
            
            info_msg = (
                f"📊 [{elapsed:.1f}s] "
                f"프레임: {self.stats['frames_processed']} ({fps:.1f} FPS), "
                f"전체얼굴: {total_faces}, "
                f"침착맨: {self.stats['chimchakman_faces']}개 ({chimchakman_rate:.1f}%), "
                f"프레임비율: {frame_face_rate:.1f}%, "
                f"감정: {self.stats['emotions_extracted']}, "
                f"탐지: {avg_detection:.3f}s, 감정: {avg_emotion:.3f}s"
            )
            
            # 얼굴 인식 통계 추가
            if self.config['face_recognition']['enabled']:
                avg_recognition = self.stats['total_recognition_time'] / max(1, self.stats['batch_count'])
                info_msg += f", 인식: {avg_recognition:.3f}s"
            
            self.logger.info(info_msg)
    
    def _print_final_stats(self):
        """최종 통계 출력 (개선된 버전)"""
        elapsed = time.time() - self.stats['processing_start_time']
        fps = self.stats['frames_processed'] / elapsed if elapsed > 0 else 0
        
        total_faces = self.stats['chimchakman_faces'] + self.stats['other_faces_filtered']
        chimchakman_rate = (self.stats['chimchakman_faces'] / max(1, total_faces)) * 100
        frame_face_rate = (self.stats['chimchakman_faces'] / max(1, self.stats['frames_processed'])) * 100
        
        self.logger.info("🎯 전처리 완료!")
        self.logger.info(f"   총 처리 시간: {elapsed:.1f}초")
        self.logger.info(f"   처리된 프레임: {self.stats['frames_processed']:,}개 ({fps:.1f} FPS)")
        self.logger.info(f"   전체 탐지된 얼굴: {total_faces:,}개")
        self.logger.info(f"   ├─ 침착맨 얼굴: {self.stats['chimchakman_faces']:,}개 ({chimchakman_rate:.1f}%)")
        self.logger.info(f"   └─ 다른 사람 얼굴: {self.stats['other_faces_filtered']:,}개")
        self.logger.info(f"   프레임당 침착맨 비율: {frame_face_rate:.1f}%")
        self.logger.info(f"   감정 분석 완료: {self.stats['emotions_extracted']:,}개")
        
        # 얼굴 이미지 저장 통계
        if self.config['video']['save_face_images']:
            self.logger.info(f"   저장된 얼굴 이미지:")
            self.logger.info(f"   ├─ 침착맨: {self.stats['chimchakman_faces']:,}개")
            self.logger.info(f"   └─ 다른 사람: {self.stats['others_faces_saved']:,}개 (최대 {self.config['video']['max_others_faces']}개)")
        
        # 평균 처리 시간
        if self.stats['batch_count'] > 0:
            avg_detection = self.stats['total_face_detection_time'] / self.stats['batch_count']
            avg_emotion = self.stats['total_emotion_time'] / self.stats['batch_count']
            self.logger.info(f"   평균 탐지 시간: {avg_detection:.3f}초/배치")
            self.logger.info(f"   평균 감정 시간: {avg_emotion:.3f}초/배치")
            
            if self.config['face_recognition']['enabled']:
                avg_recognition = self.stats['total_recognition_time'] / self.stats['batch_count']
                self.logger.info(f"   평균 인식 시간: {avg_recognition:.3f}초/배치")


def main():
    """테스트 실행"""
    import argparse
    
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='긴 영상 비디오 전처리 (개선된 버전)')
    parser.add_argument('video_path', help='처리할 비디오 파일 경로')
    parser.add_argument('--config', default='video_analyzer/configs/inference_config.yaml', help='설정 파일 경로')
    
    args = parser.parse_args()
    
    # 전처리기 실행 - video_name은 process_long_video에서 자동으로 설정됨
    processor = LongVideoProcessor(args.config)
    
    # 비디오 처리
    result = processor.process_long_video(args.video_path)
    
    if result:
        print("\n✅ 전처리 완료!")
        print(f"HDF5 파일: {result['hdf5_path']}")
        print(f"처리 시간: {result['duration']:.1f}초")
        print(f"프레임당 얼굴 탐지률: {result['face_detection_ratio']:.1%}")
        print(f"침착맨 인식률: {result['chimchakman_recognition_rate']:.1%}")
        if 'stats' in result:
            stats = result['stats']
            total_faces = stats['chimchakman_faces'] + stats['other_faces_filtered']
            print(f"전체 얼굴: {total_faces:,}개 (침착맨: {stats['chimchakman_faces']:,}개, 다른사람: {stats['other_faces_filtered']:,}개)")
        if 'face_images_dir' in result and result['face_images_dir']:
            print(f"얼굴 이미지 저장 위치: {result['face_images_dir']}")
    else:
        print("❌ 전처리 실패")


if __name__ == "__main__":
    main()