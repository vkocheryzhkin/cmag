#ifndef __RENDER_PARTICLES__
#define __RENDER_PARTICLES__

#include <GL/glew.h>

class ParticleRenderer
{
public:
    ParticleRenderer();
    ~ParticleRenderer();

    void setPositions(float *pos, int numParticles);
    void setVertexBuffer(unsigned int vbo, int numParticles);
    void setColorBuffer(unsigned int vbo) { colorVBO = vbo; }

    

    void display();
    void displayGrid();

    void setPointSize(float size)  { pointSize = size; }
    void setradius(float r) { radius = r; }
    void setFOV(float _fov) { fov = _fov; }
    void setWindowSize(int w, int h) { window_w = w; window_h = h; }

protected: // methods
    void _initGL();
    void _drawPoints();
    GLuint _compileProgram(const char *vsource, const char *fsource);

protected: // data
    float *pos;
    int numParticles;

    float pointSize;
    float radius;
    float fov;
    int window_w, window_h;

    GLuint program;

    GLuint vbo;
    GLuint colorVBO;
};

#endif //__ RENDER_PARTICLES__
