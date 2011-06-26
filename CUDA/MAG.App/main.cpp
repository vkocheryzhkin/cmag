#include <GL/glew.h>
#if defined (_WIN32)
#include <GL/wglew.h>
#endif

#include <GL/freeglut.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cuda_gl_interop.h>
#include "fluidSystem.h"
#include "poiseuilleFlowSystem.h"
#include "render_particles.h"
#include "magUtil.cuh"

#define MAX(a,b) ((a > b) ? a : b)
#define DamBreak
//#define PoiseuilleFlow

#if defined(DamBreak)
    float camera_trans[] = {0.5, 0.0, -2.5};
	DamBreakSystem *psystem = 0; 	
#elif defined(PoiseuilleFlow)
    float camera_trans[] = {0.0, 0.0, -0.0012};
	PoiseuilleFlowSystem *psystem = 0;
#endif

float camera_rot[]   = {0, 0, 0};
float camera_trans_lag[] = {0, 0, 0};
float camera_rot_lag[] = {0, 0, 0};
const float inertia = 0.1;

const int width = 1280, height = 1024;
int ox, oy;
int buttonState = 0;
bool bPause = true;
bool IsFirstTime = true; 

static int fpsCount = 0;
static int fpsLimit = 1;
unsigned int timer;

ParticleRenderer *renderer = 0;

float modelView[16];
unsigned int frameCount = 0;
void ConditionalDisplay();

void cleanup(){
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
		wglSwapIntervalEXT(0);
	}
#endif

	glEnable(GL_DEPTH_TEST);
	glClearColor(1.0, 1.0, 1.0, 1.0);

	glutReportErrors();
}

void computeFPS()
{
	frameCount++;
	fpsCount++;

	if (fpsCount == fpsLimit) {
		char fps[256];
		float ifps = 1.f / (cutGetAverageTimerValue(timer) / 1000.f);
		sprintf(fps, "Dam Break (%d particles): %3.1f fps; elapsed Time: %f",
			psystem->getNumParticles(), ifps, psystem->getElapsedTime()); 
		glutSetWindowTitle(fps);
		fpsCount = 0;         

		cutilCheckError(cutResetTimer(timer));          
	}
}


void display()
{
	cutilCheckError(cutStartTimer(timer));  
	if(IsFirstTime){
		IsFirstTime = false;
		psystem->update(); 
		if (renderer) 
			renderer->setVertexBuffer(psystem->getCurrentReadBuffer(), psystem->getNumParticles());
	}
	if (!bPause){
		psystem->update(); 
		if (renderer) 
			renderer->setVertexBuffer(psystem->getCurrentReadBuffer(), psystem->getNumParticles());
	}

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	for (int c = 0; c < 3; ++c)	{
		camera_trans_lag[c] += (camera_trans[c] - camera_trans_lag[c]) * inertia;
		camera_rot_lag[c] += (camera_rot[c] - camera_rot_lag[c]) * inertia;
	}
	glTranslatef(camera_trans_lag[0], camera_trans_lag[1], camera_trans_lag[2]);
	glRotatef(camera_rot_lag[0], 1.0, 0.0, 0.0);
	glRotatef(camera_rot_lag[1], 0.0, 1.0, 0.0);

	glGetFloatv(GL_MODELVIEW_MATRIX, modelView);

	glColor3f(0.0, 0.0, 0.0);
	//glutWireCube(2.0);

	if (renderer) renderer->display();
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
	gluPerspective(60.0, (float) w / (float) h, 0.001, 100.0);

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
		camera_trans[2] += (dy / 100.0) * 0.5 * fabs(camera_trans[2]);
	} 
	else if (buttonState & 2) {
		camera_trans[0] += dx / 100.0;
		camera_trans[1] -= dy / 100.0;
	}
	else if (buttonState & 1) {
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
	#if defined(DamBreak)
	case '2':
		psystem->changeRightBoundary();
		break;
	case '3':
		psystem->removeRightBoundary();
		break;
	#endif	
	case '\033':
	case 'q':
		exit(0);
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


void ConditionalInit()
{
		#if defined(DamBreak)
			float num = 128;
			psystem = new DamBreakSystem(
				make_uint3(num, num, 1),
				1,
				make_uint3(4 * num, 2 * num, 4),				
				1.0f / (2 * num),				
				true); 	
		#elif defined(PoiseuilleFlow)
			psystem = new PoiseuilleFlowSystem(
					make_uint3(16, 64 -  2 * 3, 1),
					3,
					make_uint3(16, 64, 4), 1.0f / (2 * (64 - 6) * 1000),
					true);
		#endif					
			
		psystem->reset();
		
		renderer = new ParticleRenderer;
		renderer->setParticleRadius(psystem->getParticleRadius());
		renderer->setColorBuffer(psystem->getColorBuffer());		

		cutilCheckError(cutCreateTimer(&timer));
}

int main(int argc, char** argv) 
{
	initGL(argc, argv);
	cudaGLInit(argc, argv);

	ConditionalInit();

	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutKeyboardFunc(key);    
	glutIdleFunc(idle);
	atexit(cleanup);
	glutMainLoop();
	if (psystem) delete psystem;
	cudaThreadExit();
}
