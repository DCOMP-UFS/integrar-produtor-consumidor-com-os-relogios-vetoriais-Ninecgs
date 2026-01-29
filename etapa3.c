/*
 * ETAPA 3 (integração)
 * Apenas adiciona:
 *  - 2 filas (recepção e envio) estilo produtor/consumidor
 *  - 2 threads extras (receptora e emissora) por processo MPI
 *
 * Compile:
 *   mpicc -Wall -Wextra -pthread etapa3.c -o etapa3
 *
 * Run:
 *   mpirun -np 3 ./etapa3
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <mpi.h>

#define N_PROC 3 
#define TAG_CLOCK 0


typedef struct {
    int v[N_PROC];
} VectorClock;

/* ---------- Funções auxiliares ---------- */

// Incrementa o relógio local de um processo (evento interno)
void localEvent(int pid, VectorClock *clock) {
    clock->v[pid]++;
}

/* ======= ADIÇÕES DA ETAPA 3: Filas + Threads ======= */

/* itens das filas */
typedef struct {
    int from;
    int vec[N_PROC];
} RecvItem;

typedef struct {
    int from;
    int to;
    int vec[N_PROC];
} SendItem;

/* fila de recepção (receptora -> central) */
#define RECVQ_SIZE 16
static RecvItem recvQ[RECVQ_SIZE];
static int recvCount = 0;

static pthread_mutex_t recvMutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t recvCondEmpty = PTHREAD_COND_INITIALIZER;
static pthread_cond_t recvCondFull  = PTHREAD_COND_INITIALIZER;

/* fila de envio (central -> emissora) */
#define SENDQ_SIZE 16
static SendItem sendQ[SENDQ_SIZE];
static int sendCount = 0;

static pthread_mutex_t sendMutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t sendCondEmpty = PTHREAD_COND_INITIALIZER;
static pthread_cond_t sendCondFull  = PTHREAD_COND_INITIALIZER;

/* flags de controle (por processo) */
static volatile int central_done = 0;
static volatile int stop_receiver = 0;
static volatile int stop_sender = 0;

/* proteção extra */
static pthread_mutex_t mpiMutex   = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t clockMutex = PTHREAD_MUTEX_INITIALIZER;

/* --- put/get da fila de recepção --- */
static void putRecv(RecvItem it) {
    pthread_mutex_lock(&recvMutex);

    while (recvCount == RECVQ_SIZE) {
        pthread_cond_wait(&recvCondFull, &recvMutex);
    }

    recvQ[recvCount++] = it;

    pthread_mutex_unlock(&recvMutex);
    pthread_cond_signal(&recvCondEmpty);
}

static int getRecv(RecvItem *out) {
    pthread_mutex_lock(&recvMutex);

    while (recvCount == 0 && !central_done) {
        pthread_cond_wait(&recvCondEmpty, &recvMutex);
    }

    if (recvCount == 0 && central_done) {
        pthread_mutex_unlock(&recvMutex);
        return 0;
    }

    *out = recvQ[0];
    for (int i = 0; i < recvCount - 1; i++) recvQ[i] = recvQ[i + 1];
    recvCount--;

    pthread_mutex_unlock(&recvMutex);
    pthread_cond_signal(&recvCondFull);
    return 1;
}

/* --- put/get da fila de envio --- */
static void putSend(SendItem it) {
    pthread_mutex_lock(&sendMutex);

    while (sendCount == SENDQ_SIZE) {
        pthread_cond_wait(&sendCondFull, &sendMutex);
    }

    sendQ[sendCount++] = it;

    pthread_mutex_unlock(&sendMutex);
    pthread_cond_signal(&sendCondEmpty);
}

static int getSend(SendItem *out) {
    pthread_mutex_lock(&sendMutex);

    while (sendCount == 0 && !stop_sender) {
        pthread_cond_wait(&sendCondEmpty, &sendMutex);
    }

    if (sendCount == 0 && stop_sender) {
        pthread_mutex_unlock(&sendMutex);
        return 0;
    }

    *out = sendQ[0];
    for (int i = 0; i < sendCount - 1; i++) sendQ[i] = sendQ[i + 1];
    sendCount--;

    pthread_mutex_unlock(&sendMutex);
    pthread_cond_signal(&sendCondFull);
    return 1;
}

/* ======= FUNÇÃO sendMsg (mesma assinatura, só mudou por dentro) ======= */
// Envia o relógio para outro processo
void sendMsg(int from, int to, VectorClock *clock) {
    pthread_mutex_lock(&clockMutex);

    clock->v[from]++;

    // em vez de MPI_Send aqui, enfileira na fila de envio
    SendItem it;
    it.from = from;
    it.to = to;
    for (int i = 0; i < N_PROC; i++)
        it.vec[i] = clock->v[i];

    printf("P%d -> P%d | Envio | Clock: (%d, %d, %d)\n",
           from, to, clock->v[0], clock->v[1], clock->v[2]);

    pthread_mutex_unlock(&clockMutex);

    putSend(it);
}

/* ======= FUNÇÃO recvMsg (mesma assinatura, só mudou por dentro) ======= */
// Recebe o relógio de outro processo e atualiza o seu
void recvMsg(int from, int self, VectorClock *clock) {
    (void)from;  

    RecvItem it;
    int ok = getRecv(&it);
    if (!ok) return;

    pthread_mutex_lock(&clockMutex);

    clock->v[self]++;

    for (int i = 0; i < N_PROC; i++) {
        if (it.vec[i] > clock->v[i])
            clock->v[i] = it.vec[i];
    }

    printf("P%d <- P%d | Recebimento | Clock: (%d, %d, %d)\n",
           self, it.from, clock->v[0], clock->v[1], clock->v[2]);

    pthread_mutex_unlock(&clockMutex);
}

/* ======= THREAD RECEPTORA (MPI_RECV -> fila recepção) ======= */
typedef struct {
    int rank;
} ThArgs;

static void* threadReceptora(void* arg) {
    ThArgs* a = (ThArgs*)arg;

    while (!stop_receiver) {
        int flag = 0;
        MPI_Status st;

        pthread_mutex_lock(&mpiMutex);
        MPI_Iprobe(MPI_ANY_SOURCE, TAG_CLOCK, MPI_COMM_WORLD, &flag, &st);
        pthread_mutex_unlock(&mpiMutex);

        if (!flag) {
            usleep(2000); // 2ms
            continue;
        }

        int msg[N_PROC];

        pthread_mutex_lock(&mpiMutex);
        MPI_Recv(msg, N_PROC, MPI_INT, st.MPI_SOURCE, TAG_CLOCK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        pthread_mutex_unlock(&mpiMutex);

        RecvItem it;
        it.from = st.MPI_SOURCE;
        for (int i = 0; i < N_PROC; i++) it.vec[i] = msg[i];

        putRecv(it);
    }

    return NULL;
}

/* ======= THREAD EMISSORA (fila envio -> MPI_SEND) ======= */
static void* threadEmissora(void* arg) {
    ThArgs* a = (ThArgs*)arg;

    while (1) {
        SendItem it;
        int ok = getSend(&it);
        if (!ok) break;

        pthread_mutex_lock(&mpiMutex);
        MPI_Send(it.vec, N_PROC, MPI_INT, it.to, TAG_CLOCK, MPI_COMM_WORLD);
        pthread_mutex_unlock(&mpiMutex);

         printf("P%d -> P%d | (Emissora) MPI_Send | Msg: (%d,%d,%d)\n",
                a->rank, it.to, it.vec[0], it.vec[1], it.vec[2]);
    }

    return NULL;
}

/* ---------- Lógica de cada processo ---------- */

void processo0() {
    VectorClock c = {{0, 0, 0}};
    printf("P0 inicial | Clock: (%d, %d, %d)\n", c.v[0], c.v[1], c.v[2]);

    localEvent(0, &c);
    printf("P0 evento interno | Clock: (%d, %d, %d)\n", c.v[0], c.v[1], c.v[2]);

    sendMsg(0, 1, &c);
    recvMsg(1, 0, &c);
    sendMsg(0, 2, &c);
    recvMsg(2, 0, &c);
    recvMsg(1, 0, &c);
    sendMsg(0, 1, &c);

    localEvent(0, &c);
    printf("P0 evento interno final | Clock: (%d, %d, %d)\n", c.v[0], c.v[1], c.v[2]);
}

void processo1() {
    VectorClock c = {{0, 0, 0}};
    printf("P1 inicial | Clock: (%d, %d, %d)\n", c.v[0], c.v[1], c.v[2]);

    sendMsg(1, 0, &c);
    recvMsg(0, 1, &c);
    recvMsg(0, 1, &c);
}

void processo2() {
    VectorClock c = {{0, 0, 0}};
    printf("P2 inicial | Clock: (%d, %d, %d)\n", c.v[0], c.v[1], c.v[2]);

    localEvent(2, &c);
    printf("P2 evento interno | Clock: (%d, %d, %d)\n", c.v[0], c.v[1], c.v[2]);

    sendMsg(2, 0, &c);
    recvMsg(0, 2, &c);
}

/* ---------- Função principal ---------- */

int main(int argc, char *argv[]) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // args threads
    ThArgs args;
    args.rank = rank;

    pthread_t thR, thS;
    pthread_create(&thR, NULL, threadReceptora, &args);
    pthread_create(&thS, NULL, threadEmissora, &args);

   
    switch (rank) {
        case 0: processo0(); break;
        case 1: processo1(); break;
        case 2: processo2(); break;
    }

    central_done = 1;

    // acorda central
    pthread_mutex_lock(&recvMutex);
    pthread_cond_broadcast(&recvCondEmpty);
    pthread_mutex_unlock(&recvMutex);

    // encerra emissora
    stop_sender = 1;
    pthread_mutex_lock(&sendMutex);
    pthread_cond_broadcast(&sendCondEmpty);
    pthread_mutex_unlock(&sendMutex);

    // encerra receptora
    stop_receiver = 1;

    pthread_join(thS, NULL);
    pthread_join(thR, NULL);

    MPI_Finalize();
    return 0;
}
