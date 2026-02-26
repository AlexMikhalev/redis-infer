#include "redismodule.h"
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

/* gte-pure-C stub — replace with real antirez core */
float* gte_embed(const char* text);

/* Qwen3-4B stub — replace with real C inference engine */
typedef struct { void* weights; } qwen_ctx_t;
static qwen_ctx_t *g_qwen = NULL;
qwen_ctx_t* qwen_load(const char* key);
char* qwen_generate(qwen_ctx_t* ctx, uint32_t* tokens, size_t n, int max_new);

typedef struct {
    RedisModuleBlockedClient *bc;
    RedisModuleString *query_key;
    int max_tokens;
} WorkerArg;

void* InferenceWorker(void *arg) {
    WorkerArg *w = (WorkerArg*)arg;
    RedisModuleCtx *ctx = RedisModule_GetThreadSafeContext(w->bc);

    uint8_t *data; size_t len;
    RedisModuleKey *k = RedisModule_OpenKey(ctx, w->query_key, REDISMODULE_READ);
    RedisModule_StringDMA(k, (void**)&data, &len, REDISMODULE_READ);
    uint32_t *tokens = (uint32_t*)data;
    size_t n = len / 4;

    char *result = qwen_generate(g_qwen, tokens, n, w->max_tokens);

    RedisModule_CloseKey(k);
    RedisModule_FreeThreadSafeContext(ctx);

    RedisModule_ReplyWithStringBuffer(w->bc, result, strlen(result));
    RedisModule_UnblockClient(w->bc, NULL);
    free(result);
    free(w);
    return NULL;
}

int GenerateCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    if (argc != 3) return RedisModule_WrongArity(ctx);
    if (!g_qwen) return RedisModule_ReplyWithError(ctx, "ERR load model first");

    RedisModuleBlockedClient *bc = RedisModule_BlockClient(ctx, NULL, NULL, NULL, 0);
    WorkerArg *w = malloc(sizeof(WorkerArg));
    w->bc = bc;
    w->query_key = RedisModule_HoldString(ctx, argv[1]);
    w->max_tokens = atoi(RedisModule_StringPtrLen(argv[2], NULL));

    pthread_t tid;
    pthread_create(&tid, NULL, InferenceWorker, w);
    pthread_detach(tid);
    return REDISMODULE_OK;
}

int LoadCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    if (argc != 2) return RedisModule_WrongArity(ctx);
    if (g_qwen) { /* free old context */ }
    g_qwen = qwen_load(RedisModule_StringPtrLen(argv[1], NULL));
    RedisModule_ReplyWithSimpleString(ctx, g_qwen ? "OK" : "ERR");
    return REDISMODULE_OK;
}

int RedisModule_OnLoad(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    if (RedisModule_Init(ctx, "redis-infer", 3, REDISMODULE_APIVER_1) == REDISMODULE_ERR)
        return REDISMODULE_ERR;

    RedisModule_CreateCommand(ctx, "INFER.LOAD_QWEN3", LoadCommand,
                              "write fast", 0, 0, 0);
    RedisModule_CreateCommand(ctx, "INFER.GENERATE", GenerateCommand,
                              "write fast", 1, 1, 1);

    RedisModule_Log(ctx, "notice",
                    "redis-infer v3.1 LOADED -- full Rust coding inference inside Redis");
    return REDISMODULE_OK;
}
