PKC MARK 벤치마크 툴

🚀 로컬 환경에서 다양한 AI 모델 LLM의 성능을 손쉽게 측정, 비교, 분석할 수 있는
      공개용 벤치마크 도구입니다.

(이 벤치마크 툴은 제가 개인용으로 로컬 AI 멀티모달 챗봇 시스템 개발 과정에서
 필요에 의해 제작하였고, 이후 공개용으로 리뉴얼한 툴입니다.)

✍️ 제작자 및 연락처
궁금한 점이나 버그 리포트, 제안 등은 아래 연락처로 편하게 문의해주십시오.

제작자: PKC
블로그: https://pkc0412.tistory.com/
이메일: pkc0412@gmail.com

📜 라이선스 (License)
본 프로젝트는 듀얼 라이선스 정책을 따릅니다.

비영리 및 오픈소스: 개인, 학술, 비영리 프로젝트에서는 GPLv3 라이선스를 따릅니다.

상업용: 본 소프트웨어를 상업적인 목적이나 폐쇄 소스 제품에 사용하고자 하는 경우,
별도의 상업용 라이선스가 필요합니다. 해당 사항은 위 이메일로 문의해주십시오.

할 수 있는 것들:

✅ 마음껏 사용하기 (개인/교육/연구)

(당신이 이 모놀리식을 분해할 **'용자'**라면...)

✅ 코드 뜯어보고 개선하기

✅ 포크해서 완전히 다른 프로젝트 만들기

✅ 오픈소스로 재배포하기 (원작자 표기 및 툴 이름은 고정)

⚠️ 주의사항 및 현실 체크
이 프로젝트는 **"완벽한 상용 소프트웨어"**가 아닙니다.

매우 특수한 환경에서는 문제가 있을 수 있음.

24시간 고객센터 운영 안 함 (개인 프로젝트니까요! 😊)

완벽하지 않아도 오픈소스 정신으로 공개함.

PKC 벤치마크 툴 - MARK

1. 개요

PKC 벤치마크 툴 - MARK는 로컬 AI 모델의 성능을 측정하기 위한 웹 기반 벤치마크 도구입니다.

GGUF(Llama), Diffusers, Transformers 등 다양한 로컬 모델을 동일한 프롬프트로 테스트하고,
VRAM 사용량, 추론 속도(TPS, TTFT), GPU 전력 및 온도 등 다양한 성능 지표를
실시간으로 비교하고 차트로 시각화할 수 있습니다.

2. 주요 기능

직관적인 웹 UI:
benchmark_canvas.html을 통해 모든 설정을 제어하고 실시간 결과를 확인합니다.

자동 모델 탐지:
config.json에 설정된 폴더 내의 모델들을 자동으로 스캔하고 분류합니다. (GGUF, Diffusers, Transformers 지원)

상세 성능 측정:
모델 로드 시간, VRAM 사용량, 첫 토큰 응답 속도(TTFT), 초당 토큰 수(TPS), GPU 전력(W), GPU 온도(°C)를
상세히 측정합니다.

파이프라인 테스트:
분석 모델(예: 감정 분석)의 결과를 언어 모델(LLM)의 프롬프트에 자동으로 주입하여 연계 테스트를 수행할 수 있습니다.

결과 이력 및 비교:
모든 테스트 결과는 브라우저(LocalStorage)에 저장되며, 과거 이력 간의 성능을 나란히 비교할 수 있습니다.

유연한 설정: 모델 캐싱, 순차/병렬 로딩, VRAM 부족 시 자동 재시도, GPU 레이어 수 등 다양한 테스트 옵션을 제공합니다.

3. 요구 사항

Python 3.11.9 (권장)

NVIDIA GPU (CUDA 12.1 환경 권장)

필수 Python 라이브러리 (설치 방법 참고)

4. 설치 방법

이 저장소의 파일들을 모두 다운로드합니다.

(권장) 터미널에서 프로젝트 폴더에 진입하여 가상 환경을 생성합니다.

python -m venv venv


생성된 가상 환경을 활성화합니다. (Windows 기준)

.\venv\Scripts\activate


(중요) PyTorch를 먼저 설치합니다.
사용자의 CUDA 버전에 맞는 PyTorch 버전을 공식 PyTorch 웹사이트에서 확인하여 설치하세요.

(예: CUDA 12.1 기준)

pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)


requirements.txt 파일에 포함된 나머지 라이브러리들을 설치합니다.

pip install -r requirements.txt


5. ⚠ 필수 설정 (가장 중요)

실행하기 전, config.json 파일을 수정하여 벤치마크할 모델이 위치한 로컬 폴더 경로를 반드시 설정해야 합니다.

config.json 파일을 텍스트 편집기로 엽니다.

models_scan_path의 값을 실제 모델들이 저장된 폴더 경로로 변경합니다.

예시 (Windows):

{
    "results_dir": "results",
    "models_scan_path": "C:/MyModels"
}


예시 (Linux/Mac):

{
    "results_dir": "results",
    "models_scan_path": "/home/user/models"
}


results_dir은 벤치마크 결과가 .json 및 .html 파일로 저장될 폴더 이름입니다. (기본값: "results")
6. 사용 방법

서버 실행: start_server_windows.bat 파일을 더블 클릭하여 실행합니다. (또는 터미널에서 python benchmark_server.py 입력)

UI 접속: 서버가 시작되면 잠시 후 자동으로 기본 웹 브라우저에서 benchmark_canvas.html 페이지가 열립니다.

설정: 테스트할 모델, 프롬프트, GPU 레이어 수, 캐싱 여부 등을 웹 UI에서 설정합니다.

실행: 'Benchmark Start' 버튼을 클릭하여 테스트를 시작합니다.

결과 확인: 'Summary'(요약), 'Log'(로그), 'Chart'(차트) 탭을 통해 실시간 결과를 확인합니다.

비교:
테스트 완료 후 'Result History'(결과 이력)에서 과거 결과를 불러오거나 'Comparison'(비교) 탭에서
여러 이력을 선택해 비교할 수 있습니다.

7. 모델 탐지 규칙

models_scan_path에 지정된 폴더 하위의 폴더들을 스캔하며, 다음과 같은 규칙으로 모델을 인식합니다.

Llama (GGUF): 폴더 내에 .gguf 파일이 있으면 llama 타입으로 등록됩니다.
(폴더 내에 mmproj 파일이 함께 있으면 VLM(CLIP) 모델로 인식)

Diffusers: 폴더 내에 .gguf 파일이 없고 model_index.json 파일이 있으면 diffusers (이미지 생성) 타입으로 등록됩니다.

Transformers: 위 2가지 조건에 맞지 않고 config.json 파일이 있으면 transformers 타입으로 등록됩니다.
(폴더 이름에 kluebert 또는 emotion이 포함되면 'analysis_model'로 자동 분류)

이메일: pkc0412@gmail.com
블로그: https://pkc0412.tistory.com/