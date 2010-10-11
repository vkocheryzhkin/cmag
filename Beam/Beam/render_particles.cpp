
#include <GL/glew.h>

#include <math.h>
#include <assert.h>
#include <stdio.h>

#include "render_particles.h"
#include "shaders.h"

#ifndef M_PI
#define M_PI    3.1415926535897932384626433832795
#endif

ParticleRenderer::ParticleRenderer()
: pos(0),
  numParticles(0),
  pointSize(1.0f),
  particleRadius(0.125f * 0.5f),
  program(0),
  vbo(0),
  colorVBO(0)
{
    _initGL();
}

ParticleRenderer::~ParticleRenderer()
{
    pos = 0;
}

void ParticleRenderer::setPositions(float *_pos, int _numParticles)
{
    pos = _pos;
    numParticles = _numParticles;
}

void ParticleRenderer::setVertexBuffer(unsigned int _vbo, int _numParticles)
{
    vbo = _vbo;
    numParticles = _numParticles;
}

void ParticleRenderer::_drawPoints()
{
    if (!vbo)
    {
        glBegin(GL_POINTS);
        {
            int k = 0;
            for (int i = 0; i < numParticles; ++i)
            {
                glVertex3fv(&pos[k]);
                k += 4;
            }
        }
        glEnd();
    }
    else
    {
        glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo);
        glVertexPointer(4, GL_FLOAT, 0, 0);
        glEnableClientState(GL_VERTEX_ARRAY);                

        if (colorVBO) {
            glBindBufferARB(GL_ARRAY_BUFFER_ARB, colorVBO);
            glColorPointer(4, GL_FLOAT, 0, 0);
            glEnableClientState(GL_COLOR_ARRAY);
        }

        glDrawArrays(GL_POINTS, 0, numParticles);

        glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
        glDisableClientState(GL_VERTEX_ARRAY); 
        glDisableClientState(GL_COLOR_ARRAY); 
    }
}

void ParticleRenderer::display()
{  
    glEnable(GL_POINT_SPRITE_ARB);
    glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
    glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);

    glUseProgram(program);
    glUniform1f( glGetUniformLocation(program, "pointScale"), window_h / tanf(fov*0.5f*(float)M_PI/180.0f) );
    glUniform1f( glGetUniformLocation(program, "pointRadius"), particleRadius );

    glColor3f(1, 1, 1);
    _drawPoints();

    glUseProgram(0);
    glDisable(GL_POINT_SPRITE_ARB);      
}

GLuint ParticleRenderer::_compileProgram(const char *vsource, const char *fsource)
{
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

    glShaderSource(vertexShader, 1, &vsource, 0);
    glShaderSource(fragmentShader, 1, &fsource, 0);
    
    glCompileShader(vertexShader);
    glCompileShader(fragmentShader);

    GLuint program = glCreateProgram();

    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);

    glLinkProgram(program);

    // check if program linked
    GLint success = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &success);

    if (!success) {
        char temp[256];
        glGetProgramInfoLog(program, 256, 0, temp);
        printf("Failed to link program:\n%s\n", temp);
        glDeleteProgram(program);
        program = 0;
    }

    return program;
}

void ParticleRenderer::_initGL()
{
    program = _compileProgram(vertexShader, spherePixelShader);

#if !defined(__APPLE__) && !defined(MACOSX)
    glClampColorARB(GL_CLAMP_VERTEX_COLOR_ARB, GL_FALSE);
    glClampColorARB(GL_CLAMP_FRAGMENT_COLOR_ARB, GL_FALSE);
#endif
}
