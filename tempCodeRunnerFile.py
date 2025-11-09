# Start the TTS worker thread
tts_thread = threading.Thread(target=tts_worker, args=(tts_queue,))
tts_thread.daemon = True # Allows main program to exit even if thread is running
tts_thread.start()
print("TTS worker thread started.")