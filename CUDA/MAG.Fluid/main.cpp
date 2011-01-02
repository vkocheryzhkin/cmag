#include <GL/glew.h>
#if defined (_WIN32)
#include <GL/wglew.h>
#endif

#include <GL/glut.h>

#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <cuda_gl_interop.h>

#include "fluidSystem.h"
#include "render_particles.h"

#define MAX_EPSILON_ERROR 5.00f
#define THRESHOLD         0.30f

#define GRID_SIZE       64
#define NUM_PARTICLES   15625

const uint width = 1024, height = 768;

int ox, oy;
int buttonState = 0;
float camera_trans[] = {0.0, 0.0, -2.2};
float camera_rot[]   = {0, 0, 0};
float camera_trans_lag[] = {0, 0, -1};
float camera_rot_lag[] = {0, 0, 0};
const float inertia = 0.1;

bool bPause = false;
uint numParticles = 0;
uint3 gridSize;

FluidSystem *psystem = 0;

static int fpsCount = 0;
static int fpsLimit = 1;
unsigned int timer;

ParticleRenderer *renderer = 0;

float modelView[16];
unsigned int frameCount = 0;

#define MAX(a,b) ((a > b) ? a : b)

extern "C" void cudaInit(int argc, char **argv);
extern "C" void cudaGLInit(int argc, char **argv);
extern "C" void copyArrayFromDevice(void* host, const void* device, unsigned int vbo, int size);

void initParticleSystem(int numParticles, uint3 gridSize, bool bUseOpenGL)
{
    psystem = new FluidSystem(numParticles, gridSize, bUseOpenGL); 
    psystem->reset();

    if (bUseOpenGL) {
        renderer = new ParticleRenderer;
        renderer->setParticleRadius(psystem->getParticleRadius());
        renderer->setColorBuffer(psystem->getColorBuffer());
    }

    cutilCheckError(cutCreateTimer(&timer));
}

void cleanup()
{
	cutilCheckError( cutDeleteTimer( timer));    
}

void initGL(int argc, char **argv)
{  
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("Fluid");

    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 GL_VERSION_1_5 GL_ARB_multitexture GL_ARB_vertex_buffer_object")) {
        fprintf(stderr, "Required OpenGL extensions missing.");
        exit(-1);
    }

#if defined (_WIN32)
    if (wglewIsSupported("WGL_EXT_swap_control")) {
        // disable vertical sync
        wglSwapIntervalEXT(0);
    }
#endif

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.25, 0.25, 0.25, 1.0);

    glutReportErrors();
}

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit) {
        char fps[256];
        float ifps = 1.f / (cutGetAverageTimerValue(timer) / 1000.f);
        sprintf(fps, "CUDA Particles (%d particles): %3.1f fps", numParticles, ifps);                  
        glutSetWindowTitle(fps);
        fpsCount = 0;         

        cutilCheckError(cutResetTimer(timer));          
    }
}

void display()
{
    cutilCheckError(cutStartTimer(timer));  

    if (!bPause)
    {
        psystem->update(); 
        if (renderer) 
            renderer->setVertexBuffer(psystem->getCurrentReadBuffer(), psystem->getNumParticles());
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    for (int c = 0; c < 3; ++c)
    {
        camera_trans_lag[c] += (camera_trans[c] - camera_trans_lag[c]) * inertia;
        camera_rot_lag[c] += (camera_rot[c] - camera_rot_lag[c]) * inertia;
    }
    glTranslatef(camera_trans_lag[0], camera_trans_lag[1], camera_trans_lag[2]);
    glRotatef(camera_rot_lag[0], 1.0, 0.0, 0.0);
    glRotatef(camera_rot_lag[1], 0.0, 1.0, 0.0);

    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);

    // cube
    //glColor3f(1.0, 1.0, 1.0);
    //glutWireCube(2.0);	
	//
	glBegin(GL_LINE_STRIP);
	float b = (powf((float) NUM_PARTICLES, 1.0f / 3.0f) * 2)/GRID_SIZE -1.0f;
	float y = 0.0f;
	glVertex3f(-1, -1, -1);
	glVertex3f(1, -1, -1);
	glVertex3f(1, 0, -1);
	glVertex3f(-1, 0, -1);
	glVertex3f(-1, -1, -1);
	glVertex3f(-1, -1, b);	
	glVertex3f(1, -1, b);
	glVertex3f(1, 0, b);
	glVertex3f(-1, 0, b);
	glVertex3f(-1, -1, b);
	glVertex3f(-1, 0, b);
	glVertex3f(-1, 0, -1);
	glVertex3f(1, 0, -1);
	glVertex3f(1, 0, b);
	glVertex3f(1, -1, b);
	glVertex3f(1, -1, -1);
	glEnd();	

    if (renderer)
        renderer->display();

    cutilCheckError(cutStopTimer(timer));  
    
    glutSwapBuffers();
    glutReportErrors();

    computeFPS();
}

inline float frand()
{
    return rand() / (float) RAND_MAX;
}

void reshape(int w, int h)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (float) w / (float) h, 0.1, 100.0);

    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, w, h);

    if (renderer) {
        renderer->setWindowSize(w, h);
        renderer->setFOV(60.0);
    }
}

void mouse(int button, int state, int x, int y)
{
    int mods;

    if (state == GLUT_DOWN)
        buttonState |= 1<<button;
    else if (state == GLUT_UP)
        buttonState = 0;

    mods = glutGetModifiers();
    if (mods & GLUT_ACTIVE_SHIFT) {
        buttonState = 2;
    } else if (mods & GLUT_ACTIVE_CTRL) {
        buttonState = 3;
    }

    ox = x; oy = y;

    glutPostRedisplay();
}

void motion(int x, int y)
{
    float dx, dy;
    dx = x - ox;
    dy = y - oy;
    if (buttonState == 3) {
        // left+middle = zoom
        camera_trans[2] += (dy / 100.0) * 0.5 * fabs(camera_trans[2]);
    } 
    else if (buttonState & 2) {
        // middle = translate
        camera_trans[0] += dx / 100.0;
        camera_trans[1] -= dy / 100.0;
    }
    else if (buttonState & 1) {
        // left = rotate
        camera_rot[0] += dy / 5.0;
        camera_rot[1] += dx / 5.0;
    }       
    ox = x; oy = y;

    glutPostRedisplay();
}

void key(unsigned char key, int , int)
{
    switch (key) 
    {
    case ' ':
        bPause = !bPause;
        break;
    case 13:
        psystem->update(); 
        if (renderer)
            renderer->setVertexBuffer(psystem->getCurrentReadBuffer(), psystem->getNumParticles());
        break;
	case '1':
		psystem->reset();
		break;
    case '\033':
    case 'q':
        exit(0);
        break;        
	case '3':		
		psystem->changeGravity();
		break;     
    }

    glutPostRedisplay();
}


void idle(void)
{    
    glutPostRedisplay();
}

void mainMenu(int i)
{
    key((unsigned char) i, 0, 0);
}

int main(int argc, char** argv) 
{
    numParticles = NUM_PARTICLES;
    uint gridDim = GRID_SIZE;

    gridSize.x = gridSize.y = gridSize.z = gridDim;
    
    initGL(argc, argv);
    cudaGLInit(argc, argv);

    initParticleSystem(numParticles, gridSize, true);
     
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutKeyboardFunc(key);    
    glutIdleFunc(idle);

    atexit(cleanup);

    glutMainLoop();
   

    if (psystem)
        delete psystem;

    cudaThreadExit();
}
