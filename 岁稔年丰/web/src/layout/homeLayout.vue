<template>
  <div class="AI">
    <div class="ai-content">
      <RouterView/>
    </div>
  </div>
</template>

<script setup lang="ts">
import AiNavBar from "@/layout/components/aiNavBar.vue";
</script>

<style scoped lang="scss">
.AI {
  width: 100%;
  min-height: 100%;
  display: flex;
  flex-direction: column;
}

.ai-content {
  width: 100%;
  margin: 0 auto;
  min-width: 1200px;
  height: calc(100% - 60px);

}

</style>
