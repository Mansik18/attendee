import io
import logging

import requests
from celery import shared_task

from bots.models import (
    AsyncTranscription,
    AsyncTranscriptionManager,
    AsyncTranscriptionStates,
    AudioChunk,
    Credentials,
    Participant,
    TranscriptionFailureReasons,
    Utterance,
)
from bots.utils import pcm_to_mp3

logger = logging.getLogger(__name__)


@shared_task(
    bind=True,
    soft_time_limit=7200,
    time_limit=7500,
)
def process_post_call_transcription(self, async_transcription_id):
    async_transcription = AsyncTranscription.objects.get(id=async_transcription_id)

    if async_transcription.state in (AsyncTranscriptionStates.COMPLETE, AsyncTranscriptionStates.FAILED):
        return

    try:
        AsyncTranscriptionManager.set_async_transcription_in_progress(async_transcription)

        recording = async_transcription.recording
        bot = recording.bot
        transcription_settings = async_transcription.transcription_settings

        # Get ElevenLabs API key
        elevenlabs_credentials_record = bot.project.credentials.filter(credential_type=Credentials.CredentialTypes.ELEVENLABS).first()
        if not elevenlabs_credentials_record:
            AsyncTranscriptionManager.set_async_transcription_failed(async_transcription, failure_data={"reason": TranscriptionFailureReasons.CREDENTIALS_NOT_FOUND})
            return

        elevenlabs_credentials = elevenlabs_credentials_record.get_credentials()
        if not elevenlabs_credentials:
            AsyncTranscriptionManager.set_async_transcription_failed(async_transcription, failure_data={"reason": TranscriptionFailureReasons.CREDENTIALS_NOT_FOUND})
            return

        api_key = elevenlabs_credentials.get("api_key")
        if not api_key:
            AsyncTranscriptionManager.set_async_transcription_failed(async_transcription, failure_data={"reason": TranscriptionFailureReasons.CREDENTIALS_NOT_FOUND, "error": "api_key not in credentials"})
            return

        # Concatenate ALL audio chunks in timestamp order (all participants mixed)
        audio_chunks = (
            AudioChunk.objects.filter(recording=recording, source=AudioChunk.Sources.PER_PARTICIPANT_AUDIO)
            .order_by("timestamp_ms")
            .iterator(chunk_size=50)
        )

        pcm_buffer = io.BytesIO()
        first_timestamp_ms = None
        total_duration_ms = 0
        sample_rate = None
        chunk_count = 0

        for audio_chunk in audio_chunks:
            if first_timestamp_ms is None:
                first_timestamp_ms = audio_chunk.timestamp_ms
                sample_rate = audio_chunk.sample_rate
            pcm_buffer.write(audio_chunk.audio_blob)
            total_duration_ms += audio_chunk.duration_ms
            chunk_count += 1

        if chunk_count == 0:
            logger.warning(f"No audio chunks found for async_transcription {async_transcription.id}")
            AsyncTranscriptionManager.set_async_transcription_complete(async_transcription)
            return

        logger.info(f"Concatenated {chunk_count} audio chunks ({total_duration_ms / 1000:.1f}s) for post-call transcription")

        pcm_data = pcm_buffer.getvalue()
        pcm_buffer.close()

        mp3_data = pcm_to_mp3(pcm_data, sample_rate=sample_rate or 32000)
        del pcm_data

        logger.info(f"Converted to MP3: {len(mp3_data)} bytes")

        # Call ElevenLabs API
        url = "https://api.elevenlabs.io/v1/speech-to-text"
        headers = {"xi-api-key": api_key}
        files = {"file": ("audio.mp3", mp3_data, "audio/mpeg")}

        data = {}
        if transcription_settings.elevenlabs_model_id():
            data["model_id"] = transcription_settings.elevenlabs_model_id()
        if transcription_settings.elevenlabs_language_code():
            data["language_code"] = transcription_settings.elevenlabs_language_code()
        data["tag_audio_events"] = transcription_settings.elevenlabs_tag_audio_events()

        response = requests.post(url, headers=headers, files=files, data=data if data else None, timeout=3600)
        del mp3_data

        if response.status_code != 200:
            logger.error(f"ElevenLabs API returned status {response.status_code}: {response.text}")
            AsyncTranscriptionManager.set_async_transcription_failed(async_transcription, failure_data={"reason": TranscriptionFailureReasons.TRANSCRIPTION_REQUEST_FAILED, "status_code": response.status_code, "response_text": response.text[:1000]})
            return

        result = response.json()
        logger.info("ElevenLabs post-call transcription completed successfully")

        transcript_text = result.get("text", "")
        words = [{"word": w.get("text"), "start": w.get("start"), "end": w.get("end")} for w in result.get("words", [])]
        transcription = {
            "transcript": transcript_text,
            "words": words,
            "language": result.get("language_code", None),
        }

        # We need a participant for the Utterance — use the bot's own participant record
        participant = Participant.objects.filter(bot=bot, is_the_bot=True).first()
        if not participant:
            participant = Participant.objects.filter(bot=bot).first()
        if not participant:
            participant = Participant.objects.create(
                bot=bot,
                uuid="post_call_transcription",
                full_name="Full Transcription",
                is_the_bot=True,
            )

        Utterance.objects.create(
            source=Utterance.Sources.PER_PARTICIPANT_AUDIO,
            recording=recording,
            async_transcription=async_transcription,
            participant=participant,
            audio_chunk=None,
            timestamp_ms=first_timestamp_ms or 0,
            duration_ms=total_duration_ms,
            transcription=transcription,
        )

        AsyncTranscriptionManager.set_async_transcription_complete(async_transcription)
        logger.info(f"Post-call transcription completed for async_transcription {async_transcription.id}")

    except Exception as e:
        logger.exception(f"Post-call transcription failed for async_transcription {async_transcription.id}: {e}")
        AsyncTranscriptionManager.set_async_transcription_failed(async_transcription, failure_data={"reason": TranscriptionFailureReasons.INTERNAL_ERROR, "error": str(e)})
