#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import yaml
import time
import json
import logging
import threading
import queue
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from facenet_pytorch import InceptionResnetV1

from mtcnn_wrapper import FaceDetector
from pytorch_classifier import TorchFacialClassifier


class TorchVideoProcessor:
    """
    PyTorch 기반 침착맨 킹받는 순간 탐지를 위한 비디오 프로세서
    얼굴 인식 기능 통합
    """
    
    def __init__(self, config_path: str = "config/config_torch.yaml"):
        """
        비디오 프로세서 초기화
        
        Args:
            config_path (str): 설정 파일 경로
        """
        # PyTorch 설정 최적화 - CUDA 컨텍스트 영향 최소화
        if torch.cuda.is_available():
            # CUDA 할당자 최적화
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            
            # PyTorch 버전에 따른 안전한 CUDA 설정
            try:
                # JIT fusion 비활성화 시도 (버전에 따라 다를 수 있음)
                if hasattr(torch, '_C'):
                    if hasattr(torch._C, '_jit_set_nvfuser_enabled'):
                        torch._C._jit_set_nvfuser_enabled(False)
                
                # CUDA 그래프 비활성화 (불필요한 최적화 방지)
                if hasattr(torch.cuda, 'graph'):
                    torch.cuda.graph.disable_compute_capability_caching()
                
                # 메모리 캐싱 정책 설정
                if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                    torch.cuda.set_per_process_memory_fraction(0.7)  # GPU 메모리의 70%만 사용
            except Exception as e:
                print(f"CUDA 최적화 설정 중 경고 (무시됨): {e}")
        
        # 설정 로드
        self.config = self._load_config(config_path)
        
        # 로깅 시스템 초기화
        self._setup_logging()
        
        # 성능 모니터링 변수
        self.stats = {
            'frames_processed': 0,
            'faces_detected': 0,
            'faces_recognized': 0,      # 새로 추가: 인식된 얼굴 수
            'faces_filtered': 0,        # 새로 추가: 필터링된 얼굴 수
            'angry_moments': 0,
            'processing_start_time': None,
            'last_stats_time': time.time(),
            'batch_count': 0,
            'total_inference_time': 0,
            'total_recognition_time': 0  # 새로 추가: 얼굴 인식 시간
        }
        
        # 종료 플래그 추가
        self.stop_flag = False
        self.face_detection_done = False
        self.classification_done = False
        
        # 큐 생성
        self.frame_queue = queue.Queue(maxsize=self.config['performance']['max_queue_size'])
        self.face_queue = queue.Queue(maxsize=self.config['performance']['max_queue_size'])
        self.result_queue = queue.Queue()
        
        # 얼굴 탐지기 초기화
        self._init_face_detector()
        
        # 얼굴 인식 모델 초기화 (옵션)
        self._init_face_recognition()
        
        # PyTorch 분류 모델 로드
        self._load_classifier()
        
        # 출력 디렉토리 생성
        self._create_output_dirs()
        
        self.logger.info("✅ TorchVideoProcessor 초기화 완료")
        self._print_config_summary()
    
    def _load_config(self, config_path: str) -> Dict:
        """설정 파일 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"✅ 설정 파일 로드: {config_path}")
            return config
        except Exception as e:
            print(f"❌ 설정 파일 로드 실패: {e}")
            raise
    
    def _setup_logging(self):
        """로깅 시스템 설정"""
        # 로거 생성
        self.logger = logging.getLogger('TorchVideoProcessor')
        self.logger.setLevel(getattr(logging, self.config['logging']['level']))
        
        # 기존 핸들러 제거
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 포매터 설정
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 파일 핸들러 (옵션)
        if self.config['logging']['save_logs']:
            log_dir = "results/video_processing/logs"
            os.makedirs(log_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"torch_processing_{timestamp}.log")
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            self.logger.info(f"로그 파일: {log_file}")
    
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
    
    def _init_face_recognition(self):
        """FaceNet 얼굴 인식 모델 초기화"""
        face_recog_config = self.config.get('face_recognition', {})
        
        # 얼굴 인식이 비활성화되었거나 테스트 모드인 경우 모델을 로드하지 않음
        if not face_recog_config.get('enabled', False) or face_recog_config.get('test_mode', False):
            mode_str = "비활성화됨" if not face_recog_config.get('enabled', False) else "테스트 모드"
            self.logger.info(f"⚠️ 얼굴 인식 {mode_str} - 모델 로드 안함")
            self.facenet_model = None
            self.target_embedding = None
            return
        
        try:
            # 명시적으로 CPU 문자열 대신 torch.device 객체 사용
            cpu_device = torch.device('cpu')
            
            # FaceNet을 명시적으로 CPU에 로드
            with torch.no_grad():
                # 먼저 기본 디바이스에 로드한 후 CPU로 이동
                self.facenet_model = InceptionResnetV1(pretrained='vggface2')
                self.facenet_model = self.facenet_model.to(cpu_device).eval()
            
            # 타겟 임베딩도 명시적으로 CPU로
            embedding_path = face_recog_config['embedding_path']
            if os.path.exists(embedding_path):
                embedding_data = np.load(embedding_path, allow_pickle=True).item()
                # numpy 배열에서 텐서로 변환 후 CPU로 이동
                self.target_embedding = torch.tensor(embedding_data['embedding']).to(cpu_device)
                
                self.logger.info(f"✅ 얼굴 인식 초기화 완료 (명시적 CPU 디바이스)")
                self.logger.info(f"   임베딩 파일: {embedding_path}")
                self.logger.info(f"   사용된 이미지: {embedding_data['num_images']}개")
                self.logger.info(f"   유사도 임계값: {face_recog_config['similarity_threshold']}")
            else:
                self.logger.error(f"❌ 임베딩 파일을 찾을 수 없습니다: {embedding_path}")
                self.facenet_model = None
                self.target_embedding = None
            
        except Exception as e:
            self.logger.error(f"❌ 얼굴 인식 초기화 실패: {e}")
            import traceback
            self.logger.error(traceback.format_exc())  # 상세 오류 스택 출력
            self.facenet_model = None
            self.target_embedding = None
    
    def _load_classifier(self):
        """PyTorch 분류 모델 로드"""
        try:
            self.classifier = TorchFacialClassifier(self.config)
            self.logger.info("✅ PyTorch 분류 모델 로드 완료")
        except Exception as e:
            self.logger.error(f"❌ 분류 모델 로드 실패: {e}")
            raise
    
    def _create_output_dirs(self):
        """출력 디렉토리 생성"""
        base_dir = self.config['output']['base_dir']
        os.makedirs(base_dir, exist_ok=True)
        
        if self.config['logging']['save_logs']:
            os.makedirs("logs", exist_ok=True)
    
    def _create_video_output_dir(self, video_path: str) -> str:
        """영상별 출력 디렉토리 생성"""
        video_name = Path(video_path).stem
        timestamp = datetime.now().strftime("%Y%m%d")
        
        # 특수문자 제거
        safe_name = "".join(c for c in video_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        
        video_dir = os.path.join(
            self.config['output']['base_dir'],
            f"{timestamp}_{safe_name}"
        )
        
        # 하위 디렉토리 생성
        if self.config['output']['save_highlights']:
            os.makedirs(os.path.join(video_dir, "highlights"), exist_ok=True)
        
        if self.config['output']['save_timestamps']:
            os.makedirs(os.path.join(video_dir, "timestamps"), exist_ok=True)
        
        if self.config['output']['save_processing_log']:
            os.makedirs(os.path.join(video_dir, "logs"), exist_ok=True)
        
        return video_dir
    
    def _print_config_summary(self):
        """설정 요약 출력"""
        self.logger.info("📋 설정 요약:")
        self.logger.info(f"   프레임 스킵: {self.config['video']['frame_skip']}프레임마다")
        self.logger.info(f"   MTCNN 배치: {self.config['mtcnn']['batch_size']}")
        
        # 얼굴 인식 설정 출력
        if self.config.get('face_recognition', {}).get('enabled', False):
            face_recog_config = self.config['face_recognition']
            self.logger.info(f"   얼굴 인식: 활성화 (임계값: {face_recog_config['similarity_threshold']})")
        else:
            self.logger.info(f"   얼굴 인식: 비활성화")
            
        self.logger.info(f"   분류 배치: {self.config['classifier']['batch_size']}")
        self.logger.info(f"   배치 타임아웃: {self.config['classifier']['batch_timeout']}초")
        self.logger.info(f"   큐 크기: {self.config['performance']['max_queue_size']}")
        self.logger.info(f"   디바이스: {self.config['classifier']['device']}")
    
    def _get_face_embeddings_batch(self, face_images: List[Image.Image]) -> torch.Tensor:
        """
        얼굴 이미지 배치에서 임베딩 추출
        
        Args:
            face_images (List[PIL.Image]): 224x224 얼굴 이미지들
            
        Returns:
            torch.Tensor: 정규화된 임베딩 벡터들 [batch_size, 512]
        """
        if self.facenet_model is None:
            return None
        
        try:
            # 224x224 → 160x160 리사이징 (FaceNet용)
            resized_images = []
            for face_img in face_images:
                resized_img = face_img.resize((160, 160), Image.BILINEAR)
                img_array = np.array(resized_img)
                img_tensor = torch.from_numpy(img_array).float().permute(2, 0, 1)
                img_tensor = (img_tensor - 127.5) / 128.0  # 정규화 [-1, 1]
                resized_images.append(img_tensor)
            
            # 배치 텐서 생성 (facenet_model과 동일한 디바이스 사용)
            batch_tensor = torch.stack(resized_images).to(self.facenet_model.device)
            
            # 임베딩 추출
            with torch.no_grad():
                embeddings = self.facenet_model(batch_tensor)
                embeddings = F.normalize(embeddings, p=2, dim=1)  # L2 정규화
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"배치 임베딩 추출 실패: {e}")
            import traceback
            self.logger.error(traceback.format_exc())  # 상세 오류 스택 출력
            return None
    
    def _calculate_similarities_batch(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        배치 임베딩과 타겟 임베딩 간의 유사도 계산
        
        Args:
            embeddings (torch.Tensor): 배치 임베딩 [batch_size, 512]
            
        Returns:
            torch.Tensor: 유사도 값들 [batch_size]
        """
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
            self.logger.error(f"유사도 계산 실패: {e}")
            import traceback
            self.logger.error(traceback.format_exc())  # 상세 오류 스택 출력
            return None
    
    def process_video(self, video_path: str) -> Dict:
        """
        비디오 파일 처리
        
        Args:
            video_path (str): 처리할 비디오 파일 경로
            
        Returns:
            Dict: 처리 결과 요약
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_path}")
        
        self.logger.info(f"🎬 비디오 처리 시작: {os.path.basename(video_path)}")
        
        # 영상별 출력 디렉토리 생성
        self.video_output_dir = self._create_video_output_dir(video_path)
        self.logger.info(f"📁 출력 디렉토리: {self.video_output_dir}")
        
        # 비디오 정보 가져오기
        video_info = self._get_video_info(video_path)
        self.logger.info(f"   길이: {video_info['duration']:.1f}초, FPS: {video_info['fps']:.1f}")
        
        # 결과 저장용 및 플래그 초기화
        self.angry_moments = []
        self.stats['processing_start_time'] = time.time()
        self.face_detection_done = False
        self.classification_done = False
        self.stop_flag = False
        
        # 스레드 시작
        threads = []
        
        # 1. 프레임 읽기 스레드
        frame_thread = threading.Thread(
            target=self._frame_reader_worker, 
            args=(video_path,)
        )
        threads.append(frame_thread)
        
        # 2. 얼굴 탐지 스레드
        face_thread = threading.Thread(target=self._face_detection_worker)
        threads.append(face_thread)
        
        # 3. 배치 분류 스레드
        classify_thread = threading.Thread(target=self._batch_classification_worker)
        threads.append(classify_thread)
        
        # 4. 성능 모니터링 스레드 (데몬으로 설정)
        monitor_thread = threading.Thread(target=self._performance_monitor)
        monitor_thread.daemon = True
        threads.append(monitor_thread)
        
        # 모든 스레드 시작
        for thread in threads:
            thread.start()
        
        # 프레임 읽기 완료 대기
        frame_thread.join()
        self.logger.info("✅ 프레임 읽기 완료")
        
        # 두 작업 모두 완료될 때까지 대기
        self.logger.info("⏳ 얼굴 탐지 및 분류 완료 대기 중...")
        while not (self.face_detection_done and self.classification_done):
            time.sleep(0.1)
        
        self.logger.info("✅ 모든 처리 완료!")
        
        # 종료 플래그 설정
        self.stop_flag = True
        
        # 워커 스레드들 정리
        face_thread.join(timeout=2)
        classify_thread.join(timeout=2)
        
        # 결과 저장
        results = self._save_results(video_path, video_info)
        
        # 최종 통계 출력
        self._print_final_stats()
        
        return results
    
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
            self.logger.info("📹 프레임 읽기 워커 종료")
    
    def _face_detection_worker(self):
        """얼굴 탐지 워커"""
        batch_size = self.config['mtcnn']['batch_size']
        frame_batch = []
        
        try:
            while True:
                frame_data = self.frame_queue.get()
                
                if frame_data is None:  # 종료 신호
                    # 남은 배치 처리
                    if frame_batch:
                        self._process_face_batch(frame_batch)
                    self.face_queue.put(None)  # 다음 워커에 종료 신호
                    break
                
                frame_batch.append(frame_data)
                
                # 배치가 찼으면 처리
                if len(frame_batch) >= batch_size:
                    self._process_face_batch(frame_batch)
                    frame_batch = []
                
                self.frame_queue.task_done()
                
        except Exception as e:
            self.logger.error(f"❌ 얼굴 탐지 오류: {e}")
        finally:
            self.face_detection_done = True
            self.logger.info("✅ 얼굴 탐지 완료")
    
    def _process_face_batch(self, frame_batch: List[Dict]):
        """프레임 배치에서 얼굴 탐지 및 인식 (MTCNN + FaceNet 배치 처리)"""
        batch_start_time = time.time()
        
        try:
            # PIL 이미지 리스트와 메타데이터 준비
            pil_images = []
            frame_metadata_list = []
            
            for frame_data in frame_batch:
                try:
                    pil_image = Image.fromarray(frame_data['frame'])
                    pil_images.append(pil_image)
                    frame_metadata_list.append({
                        'frame_number': frame_data['frame_number'],
                        'timestamp': frame_data['timestamp']
                    })
                except Exception as e:
                    self.logger.warning(f"⚠️ 프레임 {frame_data['frame_number']} PIL 변환 실패: {e}")
            
            if not pil_images:
                self.logger.warning("⚠️ 변환된 이미지가 없습니다")
                return
            
            # MTCNN 배치 처리 호출 (224x224 얼굴 이미지 추출)
            face_results = self.face_detector.process_image_batch(
                pil_images, frame_metadata_list
            )
            
            # 원본 탐지 결과의 얼굴 수 저장 (로깅용)
            total_detected = len([r for r in face_results if 'face_image' in r])
            
            # 얼굴 인식 활성화 + 모델 로드된 경우에만 필터링 수행
            # FaceNet 모델이 None이 아닌 경우에만 얼굴 인식 수행 (테스트 모드는 이미 None으로 설정됨)
            if self.facenet_model is not None:
                face_results = self._filter_faces_by_recognition(face_results)
            
            # 결과를 face_queue에 추가
            faces_in_batch = 0
            for face_data in face_results:
                try:
                    # face_queue에 추가할 데이터 구성
                    queue_data = {
                        'face_image': face_data['face_image'],
                        'frame_number': face_data['frame_number'],
                        'timestamp': face_data['timestamp']
                    }
                    self.face_queue.put(queue_data)
                    faces_in_batch += 1
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ 얼굴 데이터 큐 추가 실패: {e}")
            
            # 통계 업데이트
            self.stats['frames_processed'] += len(frame_batch)
            self.stats['faces_detected'] += total_detected
            
            batch_time = time.time() - batch_start_time
            
            # 배치 단위 로깅
            if self.config['logging']['batch_summary']:
                recognition_info = ""
                # 얼굴 인식 모델이 로드된 경우에만 필터링 정보 계산
                if self.facenet_model is not None:
                    filtered = total_detected - faces_in_batch
                    recognition_info = f" (인식 후: {faces_in_batch}개, 필터링: {filtered}개)"
                
                self.logger.info(
                    f"얼굴 탐지 배치: {len(frame_batch)}프레임 → {faces_in_batch}개 얼굴{recognition_info} "
                    f"({batch_time:.2f}초)"
                )
        
        except Exception as e:
            self.logger.error(f"❌ 얼굴 탐지 배치 처리 실패: {e}")
            # 실패 시에도 통계는 업데이트
            self.stats['frames_processed'] += len(frame_batch)
    
    def _filter_faces_by_recognition(self, face_results: List[Dict]) -> List[Dict]:
        """
        얼굴 인식으로 침착맨 얼굴만 필터링
        
        Args:
            face_results (List[Dict]): MTCNN 탐지 결과
            
        Returns:
            List[Dict]: 필터링된 얼굴 결과
        """
        # 모델이 로드되지 않았거나 결과가 없으면 필터링 없이 반환
        if not face_results or self.facenet_model is None:
            return face_results
        
        # 테스트 모드 확인 (설정 파일에서 test_mode가 true면 필터링 건너뜀)
        if self.config.get('face_recognition', {}).get('test_mode', False):
            self.logger.info("🧪 얼굴 인식 테스트 모드: 필터링 건너뜀")
            return face_results
        
        try:
            recognition_start_time = time.time()
            
            # 얼굴 이미지들 추출
            face_images = [result['face_image'] for result in face_results]
            
            # 배치 임베딩 추출
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
            recognized_count = 0
            
            for i, (result, similarity, is_match) in enumerate(zip(face_results, similarities, matches)):
                if is_match:
                    # 유사도 정보 추가
                    result['similarity'] = float(similarity)
                    filtered_results.append(result)
                    recognized_count += 1
                else:
                    # 필터링된 얼굴 저장 (옵션)
                    if self.config['face_recognition']['logging']['save_filtered_faces']:
                        self._save_filtered_face(result, float(similarity))
            
            recognition_time = time.time() - recognition_start_time
            self.stats['total_recognition_time'] += recognition_time
            self.stats['faces_recognized'] += recognized_count
            self.stats['faces_filtered'] += (len(face_results) - recognized_count)
            
            # 인식 통계 로깅
            if self.config['face_recognition']['logging']['log_filtered_count']:
                self.logger.debug(
                    f"얼굴 인식: {len(face_results)}개 → {recognized_count}개 "
                    f"(필터링: {len(face_results) - recognized_count}개, {recognition_time:.3f}초)"
                )
            
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"❌ 얼굴 인식 필터링 실패: {e}")
            import traceback
            self.logger.error(traceback.format_exc())  # 상세 오류 스택 출력
            return face_results  # 실패 시 원본 반환
    
    def _save_filtered_face(self, face_data: Dict, similarity: float):
        """필터링된 얼굴 이미지 저장 (디버깅용)"""
        try:
            timestamp_str = f"{int(face_data['timestamp']):05d}"
            filename = f"filtered_{timestamp_str}_{similarity:.3f}.jpg"
            
            filtered_dir = os.path.join(self.video_output_dir, "filtered_faces")
            os.makedirs(filtered_dir, exist_ok=True)
            
            save_path = os.path.join(filtered_dir, filename)
            face_data['face_image'].save(save_path)
            
        except Exception as e:
            self.logger.warning(f"⚠️ 필터링된 얼굴 저장 실패: {e}")
    
    def _batch_classification_worker(self):
        """배치 분류 워커"""
        batch_size = self.config['classifier']['batch_size']
        timeout = self.config['classifier']['batch_timeout']
        
        face_batch = []
        last_batch_time = time.time()
        
        try:
            while True:
                try:
                    # 타임아웃 설정으로 얼굴 데이터 가져오기
                    remaining_timeout = max(0.1, timeout - (time.time() - last_batch_time))
                    face_data = self.face_queue.get(timeout=remaining_timeout)
                    
                    if face_data is None:  # 종료 신호
                        # 남은 배치 처리
                        if face_batch:
                            self._process_classification_batch(face_batch)
                        break
                    
                    face_batch.append(face_data)
                    
                    # 배치 처리 조건 확인
                    should_process = (
                        len(face_batch) >= batch_size or  # 배치가 가득 참
                        (self.face_detection_done and len(face_batch) > 0) or  # 탐지 완료 + 남은 배치
                        (time.time() - last_batch_time) >= timeout  # 타임아웃
                    )
                    
                    if should_process:
                        self._process_classification_batch(face_batch)
                        face_batch = []
                        last_batch_time = time.time()
                    
                    self.face_queue.task_done()
                    
                except queue.Empty:
                    # 타임아웃 발생 - 현재 배치 처리
                    if face_batch:
                        if self.config['logging']['batch_summary']:
                            self.logger.info(f"타임아웃으로 배치 처리: {len(face_batch)}개")
                        self._process_classification_batch(face_batch)
                        face_batch = []
                        last_batch_time = time.time()
                
        except Exception as e:
            self.logger.error(f"❌ 분류 워커 오류: {e}")
        finally:
            self.classification_done = True
            self.logger.info("✅ 분류 처리 완료")
    
    def _process_classification_batch(self, face_batch: List[Dict]):
        """분류 배치 처리"""
        if not face_batch:
            return
        
        try:
            # 얼굴 이미지들 추출
            face_images = [face_data['face_image'] for face_data in face_batch]
            
            # 배치 예측
            batch_start_time = time.time()
            predictions = self.classifier.predict_batch(face_images)
            batch_time = time.time() - batch_start_time
            
            self.stats['batch_count'] += 1
            self.stats['total_inference_time'] += batch_time
            
            # 결과 처리
            angry_count = 0
            for face_data, prediction in zip(face_batch, predictions):
                if prediction['is_angry']:
                    angry_moment = {
                        'timestamp': face_data['timestamp'],
                        'frame_number': face_data['frame_number'],
                        'confidence': prediction['confidence']
                    }
                    
                    # 얼굴 인식 유사도 정보 추가 (있는 경우)
                    if 'similarity' in face_data:
                        angry_moment['similarity'] = face_data['similarity']
                    
                    self.angry_moments.append(angry_moment)
                    self.stats['angry_moments'] += 1
                    angry_count += 1
                    
                    # 킹받는 프레임 저장 (옵션)
                    if self.config['output']['save_highlights']:
                        self._save_highlight_image(face_data, prediction['confidence'])
                    
                    if self.config['debug']['timing_detailed']:
                        timestamp_str = str(timedelta(seconds=int(face_data['timestamp'])))
                        similarity_info = f", 유사도: {face_data.get('similarity', 'N/A'):.3f}" if 'similarity' in face_data else ""
                        self.logger.info(f"😡 킹받는 순간! {timestamp_str} (신뢰도: {prediction['confidence']:.3f}{similarity_info})")
            
        except Exception as e:
            self.logger.error(f"⚠️ 배치 분류 오류: {e}")
    
    def _save_highlight_image(self, face_data: Dict, confidence: float):
        """킹받는 순간 이미지 저장"""
        try:
            timestamp_str = f"{int(face_data['timestamp']):05d}"
            similarity_str = f"_{face_data['similarity']:.3f}" if 'similarity' in face_data else ""
            filename = f"angry_{timestamp_str}_{confidence:.3f}{similarity_str}.jpg"
            
            save_path = os.path.join(
                self.video_output_dir,
                "highlights",
                filename
            )
            
            face_data['face_image'].save(save_path)
            
        except Exception as e:
            self.logger.warning(f"⚠️ 하이라이트 이미지 저장 실패: {e}")
    
    def _performance_monitor(self):
        """성능 모니터링"""
        interval = self.config['performance']['monitoring_interval']
        
        while not self.stop_flag:
            time.sleep(interval)
            
            if self.stats['processing_start_time'] is None:
                continue
            
            # 현재 통계
            elapsed = time.time() - self.stats['processing_start_time']
            fps = self.stats['frames_processed'] / elapsed if elapsed > 0 else 0
            
            # 큐 상태
            frame_queue_size = self.frame_queue.qsize()
            face_queue_size = self.face_queue.qsize()
            
            # GPU 메모리 사용량
            memory_info = self.classifier.get_memory_usage()
            
            if self.config['logging']['performance_tracking']:
                avg_batch_time = (self.stats['total_inference_time'] / self.stats['batch_count'] 
                                 if self.stats['batch_count'] > 0 else 0)
                
                # 얼굴 인식 통계 추가 (모델이 로드된 경우에만)
                recognition_info = ""
                if self.facenet_model is not None:
                    recognition_rate = (self.stats['faces_recognized'] / max(1, self.stats['faces_detected'])) * 100
                    avg_recognition_time = (self.stats['total_recognition_time'] / max(1, self.stats['batch_count']))
                    recognition_info = f", 인식률: {recognition_rate:.1f}%, 인식시간: {avg_recognition_time:.3f}초"
                
                self.logger.info(
                    f"📊 [{elapsed:.1f}s] "
                    f"프레임: {self.stats['frames_processed']} ({fps:.1f} FPS), "
                    f"얼굴: {self.stats['faces_detected']}, "
                    f"킹받음: {self.stats['angry_moments']}, "
                    f"큐: {frame_queue_size}/{face_queue_size}, "
                    f"GPU: {memory_info['allocated']:.1f}GB, "
                    f"배치 평균: {avg_batch_time:.3f}초{recognition_info}"
                )
        
        self.logger.info("📊 성능 모니터링 종료")
    
    def _save_results(self, video_path: str, video_info: Dict) -> Dict:
        """결과 저장"""
        results = {
            'video_path': video_path,
            'video_info': video_info,
            'processing_stats': self.stats.copy(),
            'angry_moments': self.angry_moments,
            'total_angry_moments': len(self.angry_moments),
            'config': self.config
        }
        
        # 타임스탬프 JSON 저장
        if self.config['output']['save_timestamps']:
            timestamp_file = os.path.join(
                self.video_output_dir,
                "timestamps",
                "angry_moments.json"
            )
            
            with open(timestamp_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 결과 저장: {timestamp_file}")
        
        return results
    
    def _print_final_stats(self):
        """최종 통계 출력"""
        elapsed = time.time() - self.stats['processing_start_time']
        fps = self.stats['frames_processed'] / elapsed if elapsed > 0 else 0
        avg_batch_time = (self.stats['total_inference_time'] / self.stats['batch_count'] 
                         if self.stats['batch_count'] > 0 else 0)
        
        self.logger.info("🎯 처리 완료!")
        self.logger.info(f"   총 처리 시간: {elapsed:.1f}초")
        self.logger.info(f"   처리된 프레임: {self.stats['frames_processed']}개 ({fps:.1f} FPS)")
        self.logger.info(f"   탐지된 얼굴: {self.stats['faces_detected']}개")
        
        # 얼굴 인식 통계 출력 (모델이 로드된 경우에만)
        if self.facenet_model is not None:
            recognition_rate = (self.stats['faces_recognized'] / max(1, self.stats['faces_detected'])) * 100
            avg_recognition_time = (self.stats['total_recognition_time'] / max(1, self.stats['batch_count']))
            self.logger.info(f"   인식된 얼굴: {self.stats['faces_recognized']}개 ({recognition_rate:.1f}%)")
            self.logger.info(f"   필터링된 얼굴: {self.stats['faces_filtered']}개")
            self.logger.info(f"   평균 인식 시간: {avg_recognition_time:.3f}초/배치")
        
        self.logger.info(f"   킹받는 순간: {self.stats['angry_moments']}개")
        self.logger.info(f"   분류 배치: {self.stats['batch_count']}회 (평균 {avg_batch_time:.3f}초)")
        
        # GPU 메모리 최종 사용량
        memory_info = self.classifier.get_memory_usage()
        self.logger.info(f"   최대 GPU 메모리: {memory_info['max_allocated']:.1f}GB")


def main():
    """메인 실행 함수"""
    import sys
    import argparse
    
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='침착맨 킹받는 순간 탐지 (PyTorch + 얼굴 인식)')
    parser.add_argument('filename', nargs='?', help='처리할 비디오 파일명 (확장자 포함)')
    parser.add_argument('--dir', '--directory', help='비디오 파일 디렉토리 경로')
    parser.add_argument('--config', default='config/config_torch.yaml', help='설정 파일 경로')
    
    args = parser.parse_args()
    
    try:
        # 프로세서 초기화
        processor = TorchVideoProcessor(args.config)
        
        # 비디오 경로 결정
        if args.filename:
            # 명령줄에서 파일명 제공
            video_dir = args.dir if args.dir else processor.config['video']['default_directory']
            video_path = os.path.join(video_dir, args.filename)
        else:
            # config에서 기본값 사용
            video_dir = processor.config['video']['default_directory']
            video_filename = processor.config['video']['default_filename']
            video_path = os.path.join(video_dir, video_filename)
        
        processor.logger.info(f"🎬 처리할 영상: {video_path}")
        
        # 비디오 처리
        results = processor.process_video(video_path)
        
        processor.logger.info("✅ 모든 처리가 완료되었습니다!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 강제 종료
        print("🔚 프로그램을 종료합니다.")
        sys.exit(0)


if __name__ == "__main__":
    main()