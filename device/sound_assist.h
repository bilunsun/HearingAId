#ifndef SOUND_ASSIST_H
#define SOUND_ASSIST_H

#include "audio_processor.h"
#include "neural_net.h"
#include "user_out.h"

class SoundAssist {
    AudioProcessor audio_processor;
    NeuralNet neural_net;
    UserOut user_out;

  public:
    void run();
};

#endif // SOUND_ASSIST_H
