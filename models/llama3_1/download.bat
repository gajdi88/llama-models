@echo off
REM Copyright (c) Meta Platforms, Inc. and affiliates.
REM All rights reserved.
REM
REM This source code is licensed under the terms described in the LICENSE file in
REM top-level folder for each specific model found within the models/ directory at
REM the top-level of this source tree.
REM
REM Copyright (c) Meta Platforms, Inc. and affiliates.
REM This software may be used and distributed according to the terms of the Llama 3.1 Community License Agreement.

setlocal enabledelayedexpansion

REM Temporarily add wget to the PATH
set PATH=C:\Program Files (x86)\GnuWin32\bin;%PATH%

echo %PATH%

set /p PRESIGNED_URL="Enter the URL from email: "
set ALL_MODELS_LIST=meta-llama-3.1-405b,meta-llama-3.1-70b,meta-llama-3.1-8b,meta-llama-guard-3-8b,prompt-guard
echo.
echo **** Model list ****
for %%a in (%ALL_MODELS_LIST:,= %) do (
    echo -  %%a
)

set /p SELECTED_MODEL="Choose the model to download: "
echo.
echo Selected model: %SELECTED_MODEL%
echo.

set SELECTED_MODELS=

if "%SELECTED_MODEL%"=="meta-llama-3.1-405b" (
    set MODEL_LIST=meta-llama-3.1-405b-instruct-mp16,meta-llama-3.1-405b-instruct-mp8,meta-llama-3.1-405b-instruct-fp8,meta-llama-3.1-405b-mp16,meta-llama-3.1-405b-mp8,meta-llama-3.1-405b-fp8
) else if "%SELECTED_MODEL%"=="meta-llama-3.1-70b" (
    set MODEL_LIST=meta-llama-3.1-70b-instruct,meta-llama-3.1-70b
) else if "%SELECTED_MODEL%"=="meta-llama-3.1-8b" (
    set MODEL_LIST=meta-llama-3.1-8b-instruct,meta-llama-3.1-8b
) else if "%SELECTED_MODEL%"=="meta-llama-guard-3-8b" (
    set MODEL_LIST=meta-llama-guard-3-8b-int8-hf,meta-llama-guard-3-8b
) else if "%SELECTED_MODEL%"=="prompt-guard" (
    set SELECTED_MODELS=prompt-guard
    set MODEL_LIST=
)

if "%SELECTED_MODELS%"=="" (
    echo.
    echo **** Available models to download: ***
    for %%a in (%MODEL_LIST:,= %) do (
        echo -  %%a
    )
    set /p SELECTED_MODELS="Enter the list of models to download without spaces or press Enter for all: "
)

set TARGET_FOLDER="."
if not exist %TARGET_FOLDER% (
    mkdir %TARGET_FOLDER%
)

if "%SELECTED_MODELS%"=="" (
    set SELECTED_MODELS=%MODEL_LIST%
)

if "%SELECTED_MODEL%"=="meta-llama-3.1-405b" (
    echo.
    echo Model requires significant storage and computational resources, occupying approximately 750GB of disk storage space and necessitating two nodes on MP16 for inferencing.
    set /p ACK="Enter Y to continue: "
    if /i not "%ACK%"=="Y" (
        echo Exiting...
        exit /b 1
    )
)

echo Downloading LICENSE and Acceptable Usage Policy
wget --continue %PRESIGNED_URL%:*=LICENSE -O %TARGET_FOLDER%\LICENSE
wget --continue %PRESIGNED_URL%:*=USE_POLICY.md -O %TARGET_FOLDER%\USE_POLICY.md

for %%m in (%SELECTED_MODELS:,= %) do (
    set ADDITIONAL_FILES=
    set TOKENIZER_MODEL=1
    if "%%m"=="meta-llama-3.1-405b-instruct-mp16" (
        set PTH_FILE_COUNT=15
        set MODEL_PATH=Meta-Llama-3.1-405B-Instruct-MP16
    ) else if "%%m"=="meta-llama-3.1-405b-instruct-mp8" (
        set PTH_FILE_COUNT=7
        set MODEL_PATH=Meta-Llama-3.1-405B-Instruct-MP8
    ) else if "%%m"=="meta-llama-3.1-405b-instruct-fp8" (
        set PTH_FILE_COUNT=7
        set MODEL_PATH=Meta-Llama-3.1-405B-Instruct
        set ADDITIONAL_FILES=fp8_scales_0.pt,fp8_scales_1.pt,fp8_scales_2.pt,fp8_scales_3.pt,fp8_scales_4.pt,fp8_scales_5.pt,fp8_scales_6.pt,fp8_scales_7.pt
    ) else if "%%m"=="meta-llama-3.1-405b-mp16" (
        set PTH_FILE_COUNT=15
        set MODEL_PATH=Meta-Llama-3.1-405B-MP16
    ) else if "%%m"=="meta-llama-3.1-405b-mp8" (
        set PTH_FILE_COUNT=7
        set MODEL_PATH=Meta-Llama-3.1-405B-MP8
    ) else if "%%m"=="meta-llama-3.1-405b-fp8" (
        set PTH_FILE_COUNT=7
        set MODEL_PATH=Meta-Llama-3.1-405B
    ) else if "%%m"=="meta-llama-3.1-70b-instruct" (
        set PTH_FILE_COUNT=7
        set MODEL_PATH=Meta-Llama-3.1-70B-Instruct
    ) else if "%%m"=="meta-llama-3.1-70b" (
        set PTH_FILE_COUNT=7
        set MODEL_PATH=Meta-Llama-3.1-70B
    ) else if "%%m"=="meta-llama-3.1-8b-instruct" (
        set PTH_FILE_COUNT=0
        set MODEL_PATH=Meta-Llama-3.1-8B-Instruct
    ) else if "%%m"=="meta-llama-3.1-8b" (
        set PTH_FILE_COUNT=0
        set MODEL_PATH=Meta-Llama-3.1-8B
    ) else if "%%m"=="meta-llama-guard-3-8b-int8-hf" (
        set PTH_FILE_COUNT=-1
        set MODEL_PATH=Meta-Llama-Guard-3-8B-INT8-HF
        set ADDITIONAL_FILES=generation_config.json,model-00001-of-00002.safetensors,model-00002-of-00002.safetensors,model.safetensors.index.json,special_tokens_map.json,tokenizer_config.json,tokenizer.json
        set TOKENIZER_MODEL=0
    ) else if "%%m"=="meta-llama-guard-3-8b" (
        set PTH_FILE_COUNT=0
        set MODEL_PATH=Meta-Llama-Guard-3-8B
    ) else if "%%m"=="prompt-guard" (
        set PTH_FILE_COUNT=-1
        set MODEL_PATH=Prompt-Guard
        set ADDITIONAL_FILES=model.safetensors,special_tokens_map.json,tokenizer_config.json,tokenizer.json
        set TOKENIZER_MODEL=0
    )

    echo.
    echo ***Downloading !MODEL_PATH!***
    if not exist %TARGET_FOLDER%\!MODEL_PATH! (
        mkdir %TARGET_FOLDER%\!MODEL_PATH!
    )

    if "!TOKENIZER_MODEL!"=="1" (
        echo Downloading tokenizer
        wget --continue %PRESIGNED_URL:*=!MODEL_PATH!/tokenizer.model% -O %TARGET_FOLDER%\!MODEL_PATH!\tokenizer.model
    )

    if !PTH_FILE_COUNT! geq 0 (
        for /l %%s in (0, 1, !PTH_FILE_COUNT!) do (
            set s=00%%s
            set s=!s:~-2!
            echo Downloading consolidated.!s!.pth
            wget --continue %PRESIGNED_URL:*=!MODEL_PATH!/consolidated.!s!.pth% -O %TARGET_FOLDER%\!MODEL_PATH!\consolidated.!s!.pth
        )
    )

    for %%f in (!ADDITIONAL_FILES:,= !) do (
        echo Downloading %%f...
        wget --continue %PRESIGNED_URL:*=!MODEL_PATH!/%%f% -O %TARGET_FOLDER%\!MODEL_PATH!\%%f
    )

    if not "%%m"=="prompt-guard" if not "%%m"=="meta-llama-guard-3-8b-int8-hf" (
        echo Downloading params.json...
        wget --continue %PRESIGNED_URL:*=!MODEL_PATH!/params.json% -O %TARGET_FOLDER%\!MODEL_PATH!\params.json
    )
)