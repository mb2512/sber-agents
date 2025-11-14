import logging
import tempfile
import os
from pathlib import Path
from typing import Optional
from openai import AsyncOpenAI, APIError
from config import config

voice_logger = logging.getLogger("voice")

class VoiceTranscription:
    """Результат транскрибации голосового сообщения."""
    def __init__(
        self,
        text: str,
        confidence: float = 1.0,
        language: Optional[str] = None,
        duration_seconds: float = 0.0,
        raw_transcript: Optional[dict] = None
    ):
        self.text = text
        self.confidence = confidence
        self.language = language
        self.duration_seconds = duration_seconds
        self.raw_transcript = raw_transcript or {}


class VoiceTranscriptionService:
    """Сервис для транскрибации голосовых сообщений через OpenAI Whisper API или локальную модель."""
    
    def __init__(self):
        self.stt_provider = config.STT_PROVIDER
        self.whisper_model = config.WHISPER_MODEL
        
        if self.stt_provider == "openai_whisper":
            # Используем OpenAI API для Whisper
            whisper_api_key = config.OPENAI_WHISPER_API_KEY or config.OPENAI_API_KEY
            whisper_base_url = config.WHISPER_BASE_URL
            
            self.whisper_client = AsyncOpenAI(
                api_key=whisper_api_key,
                base_url=whisper_base_url
            )
        else:
            # Для локальной модели клиент не нужен
            self.whisper_client = None
    
    async def transcribe_voice_message(
        self,
        audio_file_path: str,
        language: Optional[str] = "ru"
    ) -> VoiceTranscription:
        """
        Транскрибирует голосовое сообщение в текст.
        
        Args:
            audio_file_path: Путь к аудиофайлу
            language: Язык аудио (опционально, по умолчанию 'ru' для русского)
        
        Returns:
            VoiceTranscription с результатами транскрибации
        """
        import time
        start_time = time.time()
        
        try:
            voice_logger.info(f"Starting transcription for file: {audio_file_path}, provider: {self.stt_provider}")
            
            # Проверяем размер файла
            file_size = os.path.getsize(audio_file_path)
            file_size_mb = file_size / (1024 * 1024)
            voice_logger.info(f"Audio file size: {file_size_mb:.2f} MB")
            
            if self.stt_provider == "openai_whisper" and self.whisper_client:
                # Используем OpenAI Whisper API
                with open(audio_file_path, "rb") as audio_file:
                    transcript = await self.whisper_client.audio.transcriptions.create(
                        model=self.whisper_model,
                        file=audio_file,
                        language=language,
                        response_format="verbose_json"
                    )
                
                text = transcript.text
                duration = getattr(transcript, 'duration', 0.0)
                detected_language = getattr(transcript, 'language', language)
                # Whisper API не возвращает confidence напрямую
                confidence = 1.0 if text and len(text.strip()) > 0 else 0.0
            elif self.stt_provider == "whisper_local":
                # Используем локальную модель Whisper
                try:
                    import whisper
                except ImportError:
                    raise ImportError(
                        "Библиотека 'openai-whisper' не установлена. "
                        "Установите её: pip install openai-whisper"
                    )
                
                voice_logger.info(f"Loading local Whisper model: {self.whisper_model}")
                model = whisper.load_model(self.whisper_model)
                
                voice_logger.info("Transcribing with local Whisper model...")
                result = model.transcribe(audio_file_path, language=language)
                
                text = result["text"].strip()
                detected_language = result.get("language", language)
                duration = result.get("duration", 0.0)
                
                # Для локальной модели confidence можно получить из segments
                segments = result.get("segments", [])
                if segments:
                    # Используем среднюю вероятность из сегментов
                    avg_prob = sum(s.get("no_speech_prob", 0) for s in segments) / len(segments)
                    confidence = max(0.0, 1.0 - avg_prob)  # Инвертируем no_speech_prob
                else:
                    confidence = 1.0 if text else 0.0
            else:
                raise NotImplementedError(
                    f"STT provider '{self.stt_provider}' не поддерживается. "
                    "Используйте 'openai_whisper' или 'whisper_local'."
                )
            
            processing_time = time.time() - start_time
            
            voice_logger.info(
                f"Transcription completed: text_length={len(text)}, "
                f"duration={duration:.2f}s, processing_time={processing_time:.2f}s, "
                f"language={detected_language}"
            )
            
            return VoiceTranscription(
                text=text,
                confidence=confidence,
                language=detected_language,
                duration_seconds=duration,
                raw_transcript={
                    "text": text,
                    "language": detected_language,
                    "duration": duration
                }
            )
            
        except APIError as e:
            voice_logger.error(f"OpenAI API error during transcription: {e}")
            # Если ошибка связана с регионом, предлагаем использовать локальную модель
            if "403" in str(e) or "unsupported_country" in str(e).lower():
                raise Exception(
                    "OpenAI API недоступен в вашем регионе. "
                    "Пожалуйста, используйте локальную модель Whisper или другой STT провайдер."
                )
            raise
        except Exception as e:
            voice_logger.error(f"Error during transcription: {e}", exc_info=True)
            raise


async def download_voice_file(bot, voice_file_id: str) -> str:
    """
    Скачивает голосовой файл из Telegram во временный файл.
    
    Args:
        bot: Экземпляр aiogram Bot
        voice_file_id: ID файла в Telegram
    
    Returns:
        Путь к временному файлу
    """
    try:
        # Получаем информацию о файле
        file_info = await bot.get_file(voice_file_id)
        
        # Создаем временный файл
        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".ogg",
            prefix="voice_"
        )
        temp_path = temp_file.name
        temp_file.close()
        
        # Скачиваем файл
        await bot.download_file(file_info.file_path, destination=temp_path)
        
        voice_logger.info(f"Voice file downloaded to: {temp_path}, size: {os.path.getsize(temp_path)} bytes")
        
        return temp_path
        
    except Exception as e:
        voice_logger.error(f"Error downloading voice file: {e}", exc_info=True)
        raise

